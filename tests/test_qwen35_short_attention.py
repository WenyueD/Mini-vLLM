import pytest
import torch
from myvllm.models.qwen35_attention import short_cached_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton")


@pytest.mark.parametrize("context,dim,heads,kv_heads", [(32,4,2,1),(257,256,24,4),(4299,256,24,4)])
@pytest.mark.parametrize("query_length", [3,8])
@torch.inference_mode()
def test_short_attention_causal_gqa_and_exact_serial_rows(context,dim,heads,kv_heads,query_length):
    torch.manual_seed(42)
    q=torch.randn(4,query_length,heads,dim,device="cuda",dtype=torch.bfloat16)
    prefixes=[context-query_length,context-query_length-4,context//2,0]
    kv=[(torch.randn(1,p+query_length,kv_heads,dim,device="cuda",dtype=q.dtype),
         torch.randn(1,p+query_length,kv_heads,dim,device="cuda",dtype=q.dtype)) for p in prefixes]
    originals=[(k.clone(),v.clone()) for k,v in kv]
    actual=short_cached_attention(q,kv,prefixes,context)
    serial=[]
    for pos in range(query_length):
        rows=[(k[:,:p+pos+1],v[:,:p+pos+1]) for (k,v),p in zip(kv,prefixes)]
        serial.append(short_cached_attention(q[:,pos:pos+1],rows,[p+pos for p in prefixes],context))
    torch.testing.assert_close(actual,torch.cat(serial,dim=1),rtol=0,atol=0)
    reference=[]
    for row,((k,v),prefix) in enumerate(zip(kv,prefixes)):
        key=k[0].repeat_interleave(heads//kv_heads,dim=1).transpose(0,1).float()
        value=v[0].repeat_interleave(heads//kv_heads,dim=1).transpose(0,1).float()
        scores=(q[row].transpose(0,1).float()@key.transpose(-1,-2)).bfloat16()
        scores=(scores*dim**-0.5).bfloat16().float()
        allowed=torch.arange(k.shape[1],device="cuda")[None,:] <= prefix+torch.arange(query_length,device="cuda")[:,None]
        probs=scores.masked_fill(~allowed[None,:,:],-float("inf")).softmax(-1).bfloat16().float()
        reference.append((probs@value).transpose(0,1).bfloat16())
    torch.testing.assert_close(actual.float(),torch.stack(reference).float(),rtol=.02,atol=.02)
    for (key,value),(old_key,old_value) in zip(kv,originals):
        torch.testing.assert_close(key,old_key,rtol=0,atol=0)
        torch.testing.assert_close(value,old_value,rtol=0,atol=0)
