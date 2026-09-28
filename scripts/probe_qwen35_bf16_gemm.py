"""Check whether batched token GEMMs reproduce cached single-token BF16 GEMMs."""

import torch
import torch.nn.functional as F


def time_cuda(fn, repeats=5):
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return min(times)


@torch.inference_mode()
def probe(batch: int, tokens: int, in_features: int, out_features: int) -> None:
    torch.manual_seed(20260928)
    x = torch.randn(batch, tokens, in_features, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16)
    serial = torch.stack(
        [
            F.linear(x[:, index].contiguous().unsqueeze(1), weight)[:, 0]
            for index in range(tokens)
        ],
        dim=1,
    )
    flattened = F.linear(x, weight)
    batched = torch.bmm(
        x.transpose(0, 1).contiguous(),
        weight.T.expand(tokens, -1, -1),
    ).transpose(0, 1)
    streams = [torch.cuda.Stream() for _ in range(tokens)]

    def streamed_linear():
        current = torch.cuda.current_stream()
        outputs = []
        for index, stream in enumerate(streams):
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                outputs.append(F.linear(
                    x[:, index].contiguous().unsqueeze(1), weight,
                ))
        for stream in streams:
            current.wait_stream(stream)
        return torch.cat(outputs, dim=1)

    streamed = streamed_linear()
    for name, result in (("flattened", flattened), ("batched", batched), ("streamed", streamed)):
        diff = (serial.float() - result.float()).abs()
        print(
            f"B={batch} T={tokens} {in_features}->{out_features} {name}: "
            f"different={int(torch.count_nonzero(serial != result))}/{serial.numel()} "
            f"max={diff.max().item():.7f} mean={diff.mean().item():.7f}",
            flush=True,
        )
    print(
        f"  ms serial={time_cuda(lambda: [F.linear(x[:, i].contiguous().unsqueeze(1), weight) for i in range(tokens)]):.3f} "
        f"streamed={time_cuda(streamed_linear):.3f} "
        f"flattened={time_cuda(lambda: F.linear(x, weight)):.3f}",
        flush=True,
    )
    del x, weight, serial, flattened, batched, streamed


def main() -> None:
    for batch in (2,):
        for in_features, out_features in (
            (5120, 5120), (5120, 17408), (17408, 5120),
            (5120, 10240), (5120, 6144), (5120, 48),
            (5120, 12288), (5120, 1024), (6144, 5120),
        ):
            probe(batch, 8, in_features, out_features)


if __name__ == "__main__":
    main()
