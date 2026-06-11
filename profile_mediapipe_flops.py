"""
Compute total GFLOPs of the MediaPipe Hands pipeline by walking the TFLite
flatbuffers inside hand_landmarker.task. Writes the result into
eval_results_cpu/mediapipe_hands_both/summary.json.

MediaPipe Hands runs:
  - hand_detector (palm)          — every frame when tracking is lost
  - hand_landmarks_detector       — every frame
We sum both: this is the upper-bound steady-state per-frame compute, matching
what evaluate_mediapipe.py timed end-to-end.

FLOP accounting per op (matches the Hennessy/Patterson / Han 2015 convention
of counting a MAC as 2 FLOPs):
  Conv2D            : 2 * H_o * W_o * K_h * K_w * C_in  * C_out
  DepthwiseConv2D   : 2 * H_o * W_o * K_h * K_w *         C_out * multiplier
  FullyConnected    : 2 * N_in * N_out
  TransposeConv     : 2 * H_o * W_o * K_h * K_w * C_in  * C_out
Element-wise / pooling / softmax / reshape: << 1% of total -> ignored.
"""
import json
import os
import zipfile
import io

import tflite                                  # pure-Python TFLite schema bindings


TASK = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'hand_landmarker.task')
OUT_SUMMARY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'eval_results_cpu', 'mediapipe_hands_both', 'summary.json')


def _shape(tensor):
    # tensor.ShapeAsNumpy() returns numpy array of dims; falls back to ShapeLength()
    s = tensor.ShapeAsNumpy()
    return [int(x) for x in s] if hasattr(s, '__iter__') else []


def flops_of_model(buf):
    model = tflite.Model.GetRootAsModel(buf, 0)
    sub   = model.Subgraphs(0)               # single subgraph for all MP models
    op_code_table = [model.OperatorCodes(i).BuiltinCode()
                     for i in range(model.OperatorCodesLength())]

    total = 0
    breakdown = {}
    for i in range(sub.OperatorsLength()):
        op   = sub.Operators(i)
        code = op_code_table[op.OpcodeIndex()]
        # Output tensor 0 gives us H_out / W_out / C_out
        out_t = sub.Tensors(op.Outputs(0))
        out_shape = _shape(out_t)
        # Input tensor 0 = activations (kernel weights for conv are input 1)
        in_t = sub.Tensors(op.Inputs(0))
        in_shape  = _shape(in_t)

        f = 0
        op_name = ''
        # CONV_2D = 3, DEPTHWISE_CONV_2D = 4, FULLY_CONNECTED = 9,
        # TRANSPOSE_CONV = 67. (See tflite.BuiltinOperator.)
        if code == tflite.BuiltinOperator.CONV_2D and op.InputsLength() >= 2:
            w_t = sub.Tensors(op.Inputs(1))
            w_shape = _shape(w_t)            # [C_out, K_h, K_w, C_in]
            if len(out_shape) == 4 and len(w_shape) == 4:
                _, H_o, W_o, C_out = out_shape
                _, K_h, K_w, C_in  = w_shape
                f = 2 * H_o * W_o * K_h * K_w * C_in * C_out
                op_name = 'Conv2D'
        elif code == tflite.BuiltinOperator.DEPTHWISE_CONV_2D and op.InputsLength() >= 2:
            w_t = sub.Tensors(op.Inputs(1))
            w_shape = _shape(w_t)            # [1, K_h, K_w, C_out]
            if len(out_shape) == 4 and len(w_shape) == 4:
                _, H_o, W_o, C_out = out_shape
                _, K_h, K_w, _     = w_shape
                f = 2 * H_o * W_o * K_h * K_w * C_out
                op_name = 'DepthwiseConv2D'
        elif code == tflite.BuiltinOperator.FULLY_CONNECTED and op.InputsLength() >= 2:
            w_t = sub.Tensors(op.Inputs(1))
            w_shape = _shape(w_t)            # [N_out, N_in]
            if len(w_shape) == 2:
                N_out, N_in = w_shape
                f = 2 * N_in * N_out
                op_name = 'FullyConnected'
        elif code == tflite.BuiltinOperator.TRANSPOSE_CONV and op.InputsLength() >= 3:
            w_t = sub.Tensors(op.Inputs(1))
            w_shape = _shape(w_t)            # [C_out, K_h, K_w, C_in]
            if len(out_shape) == 4 and len(w_shape) == 4:
                _, H_o, W_o, C_out = out_shape
                _, K_h, K_w, C_in  = w_shape
                f = 2 * H_o * W_o * K_h * K_w * C_in * C_out
                op_name = 'TransposeConv'

        if f:
            total += f
            breakdown[op_name] = breakdown.get(op_name, 0) + f

    return total, breakdown


def main():
    if not os.path.isfile(TASK):
        raise SystemExit(f'No hand_landmarker.task at {TASK}')

    totals = {}
    with zipfile.ZipFile(TASK) as z:
        for name in z.namelist():
            if not name.endswith('.tflite'):
                continue
            buf = z.read(name)
            f, br = flops_of_model(buf)
            gf = f / 1e9
            totals[name] = (gf, br)
            print(f'{name:35s}  {gf:7.4f} GFLOPs')
            for k, v in sorted(br.items(), key=lambda x: -x[1]):
                print(f'    {k:18s}  {v/1e9:7.4f}')

    total_gflops = sum(gf for gf, _ in totals.values())
    print(f'\nTOTAL  {total_gflops:.4f} GFLOPs per frame '
          f'(palm detector + hand landmark)')

    # Patch the MediaPipe summary.json.
    if os.path.isfile(OUT_SUMMARY):
        with open(OUT_SUMMARY, 'r', encoding='utf-8') as f:
            d = json.load(f)
        d['compute'] = {
            'gflops_forward': round(total_gflops, 4),
            'gflops_note': 'TFLite ops profiled from hand_landmarker.task '
                           '(palm detector + hand landmark, 1 MAC = 2 FLOPs)',
        }
        with open(OUT_SUMMARY, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=2)
        print(f'\npatched {OUT_SUMMARY}')


if __name__ == '__main__':
    main()
