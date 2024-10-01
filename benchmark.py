import sys
import torch
import torch.nn as nn

from model import CustomTransformerModel

V = 6000  # vocab size
model_params = {
    "ntoken": V,
    #"ninp": 256,
    "ninp": 512,
    #"nhid": 1024,
    "nhid": 2024,
    "nhead": 8,
    "nlayers": 12,
    "dropout": 0.1,
}
use_gpu = torch.cuda.is_available()
device = torch.device("cuda") if use_gpu else torch.device("cpu")
criterion = nn.NLLLoss()
train = False


def build_model(attention_type, **params):
    model = CustomTransformerModel(attention_type=attention_type, **params)
    return model.to(device)


def build_sequence(length):
    XY = torch.randint(V, (1, length + 1)).to(device)
    return XY[:, :length], XY[:, 1:length+1]


def train_step(model, X, Y):
    # Simulate training
    model.train()
    model.zero_grad()
    out = model(X)
    loss = criterion(out.view(-1, V), Y.view(-1))
    loss.backward()
    # model update
    for p in model.parameters():
        if p.grad is not None:
            p.data.add_(p.grad, alpha=2e-3)
    # cuda operations are asynchronous
    # we need to synchronize GPU and CPU, e.g. to mesure time complexity
    torch.cuda.synchronize()
    del out
    del loss
    return


def eval_step(model, X, Y):
    model.eval()
    with torch.no_grad():
        out = model(X)
        loss = criterion(out.view(-1, V), Y.view(-1))
    torch.cuda.synchronize()
    del out
    del loss
    return


def benchmark_mem(model, trials=10, min_len=1000, max_len=100_000, step=1000, train=True):
    report = Report("Peak GPU Memory", index_col_name="sequence_length")
    state_dict = model.state_dict()
    for length in range(min_len, max_len, step):
        use_gpu and torch.cuda.reset_max_memory_allocated()

        for trial in range(trials):
            X, Y = build_sequence(length)
            try:
                if train:
                    train_step(model, X, Y)
                else:
                    eval_step(model, X, Y)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory')
                    torch.cuda.empty_cache()
                    break
                else:
                    raise e
            del X
            del Y

        mem = torch.cuda.max_memory_allocated()
        report.log(sequence_length=length, gpu_memory=mem)
        model.load_state_dict(state_dict)
    return report


def benchmark_all():
    mem_report = Report("Peak GPU Memory", "sequence_length")
    for attention_type in ["euclidean", "fast_euclidean", "hyper_mixing", "summary_mixing", "full"]:
        print(f"Memory usage of {attention_type} model...")
        model = build_model(attention_type, **model_params)
        r = benchmark_mem(model, train=train)
        r.name = attention_type
        if not mem_report.has_col("sequence_length"):
            mem_report.report["sequence_length"] = r.report["sequence_length"]
        r.drop_col("sequence_length")
        mem_report = mem_report.merge(r)
        del model
    mem_report.to_csv()
    print("Saving results into `benchmark.csv`")
    mem_report.to_csv("benchmark.csv")


class Report:
    def __init__(self, name, index_col_name):
        self.name = name
        self.index_col_name = index_col_name
        self.report = dict()

    def log(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.report:
                self.report[k].append(v)
            else:
                self.report[k] = [v]

    def merge(self, other):
        for k in other.report.keys():
            nk = f"{other.name}.{k}"
            self.report[nk] = other.report[k]
        return self

    def drop_col(self, name):
        del self.report[name]
        return self

    def get_col(self, name):
        return self.report[name]

    def has_col(self, name):
        return name in self.report

    def __str__(self):
        res = f"Report: {self.name}\n"
        res += str(self.report)
        return res

    def to_csv(self, filename="/dev/stdout"):
        y_columns = [k for k in self.report.keys() if k != self.index_col_name]
        f = open(filename, "w")
        print(self.index_col_name + "," + ",".join(y_columns), file=f)
        for d in zip(self.report[self.index_col_name], *[self.report[col] for col in y_columns]):
            print(",".join(map(str, d)), file=f)
        if filename is not None:
            f.close()


if __name__ == "__main__":
    benchmark_all()
