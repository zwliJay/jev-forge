"""Optional CUDA bindings for the recorded Torch 2.8 / Transformers 5.19 run.

Install causal-conv1d and flash-linear-attention separately. Call before loading
Qwen3.5. This is an explicit runtime integration, not an architecture change.
The full-path, right-padded candidate encoder uses no decoding conv state.
"""
def enable_native_kernels():
    import importlib.metadata
    if importlib.metadata.version("transformers") != "5.19.0":
        raise RuntimeError("These explicit kernel bindings require Transformers 5.19.0")
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    def convolution(x, weight, bias=None, activation=None, **kwargs):
        return causal_conv1d_fn(x, weight, bias, activation=activation)
    def convolution_update(x, conv_state, weight, bias=None, activation=None, **kwargs):
        return causal_conv1d_update(x, conv_state, weight, bias, activation=activation)
    q.causal_conv1d_fn=convolution
    q.causal_conv1d_update=convolution_update
    q.torch_chunk_gated_delta_rule=chunk_gated_delta_rule
    q.torch_recurrent_gated_delta_rule=fused_recurrent_gated_delta_rule
    return {'convolution_backend':'causal-conv1d-cuda','delta_rule_backend':chunk_gated_delta_rule.__module__,'hub_kernels':False}
