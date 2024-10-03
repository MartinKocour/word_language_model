import sys
import torch
import torch.nn as nn

from model import CustomTransformerModel
from torch.profiler import profile, record_function, ProfilerActivity

B = 4
V = 128  # vocab size
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
causal = True


def build_model(attention_type, **params):
    model = CustomTransformerModel(attention_type=attention_type, **params)
    return model.to(device)


def build_sequence(length):
    # B = 2**13 // length or 1
    XY = torch.randint(V, (B, length + 1)).to(device)
    return XY[:, :length], XY[:, 1:length+1]


def train_step(model, X, Y):
    # Simulate training
    model.train()
    model.zero_grad()
    out = model(X, has_mask=causal)
    loss = criterion(out.reshape(-1, V), Y.reshape(-1))
    loss.backward()
    # model update
    for p in model.parameters():
        if p.grad is not None:
            p.data.add_(p.grad, alpha=2e-3)
    # cuda operations are asynchronous
    # we need to synchronize GPU and CPU, e.g. to mesure time complexity
    use_gpu and torch.cuda.synchronize()
    return


def eval_step(model, X, Y):
    model.eval()
    with torch.no_grad():
        out = model(X, has_mask=causal)
        loss = criterion(out.reshape(-1, V), Y.reshape(-1))
    use_gpu and torch.cuda.synchronize()
    return loss


def benchmark(model, train=True):
    report = Report("Peak GPU Memory", index_col_name="sequence_length")
    state_dict = model.state_dict()
    for length in map(int, torch.logspace(6, 14, 9, base=2)):
        use_gpu and torch.cuda.reset_max_memory_allocated()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            with_flops=True,
            # record_shapes=True,
            # run profiling after 5 steps for another 5 steps (due to torch initialization etc.)
            schedule=torch.profiler.schedule(wait=2, warmup=1, active=5, repeat=0, skip_first=2)
        ) as prof:
            for trial in range(10):
                X, Y = build_sequence(length)
                try:
                    if train:
                        train_step(model, X, Y)
                    else:
                        eval_step(model, X, Y)

                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print(f'| WARNING: ran out of memory with input of shape {X.shape}')
                        torch.cuda.empty_cache()
                        return report
                    else:
                        raise e
                prof.step()

        # B = 2**16//length
        # `Attention Layer` is defined in fast_transformers/attention/attention_layer.py
        prof_stats = list(filter(lambda k: k.key == "Attention Layer", prof.key_averages()))[0]
        prof_norm = B * 5
        cuda_mem = prof_stats.device_memory_usage / prof_norm  # bytes
        cpu_mem = prof_stats.cpu_memory_usage / prof_norm      # bytes
        cpu_time = prof_stats.cpu_time_total / prof_norm       # micro seconds
        cuda_time = prof_stats.device_time_total / prof_norm   # micro seconds
        report.log(sequence_length=length, gpu_memory=cuda_mem, cuda_time=cuda_time, cpu_memory=cpu_mem, cpu_time=cpu_time)
        model.load_state_dict(state_dict)
        # print(
        #     prof.key_averages().table(
        #         # sort_by="cpu_time_total",
        #         sort_by="flops",
        #         row_limit=-1,
        #         # top_level_events_only=True,
        #         top_level_events_only=False,
        #     )
        # )
    return report


def benchmark_all():
    report = Report("Peak GPU Memory", "sequence_length")
    for attention_type in ["euclidean", "fast_euclidean", "hyper_mixing", "summary_mixing", "full", "linear"]:
        if causal and attention_type == "linear":
            continue
        print(f"Benchmarking {attention_type} model...")
        model = build_model(attention_type, **model_params)
        r = benchmark(model, train=train)
        r.name = attention_type
        if not report.has_col("sequence_length"):
            report.report["sequence_length"] = r.report["sequence_length"]
        r.drop_col("sequence_length")
        report = report.merge(r)
        del model
    report.to_csv()
    print("Saving results into `benchmark.csv`")
    report.to_csv("benchmark.csv")


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
