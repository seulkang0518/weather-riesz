import torch
from weather.uniform_riesz import uniform_toward, uniform_riesz_loss
from weather.riesz_topk_softmax import _riesz_topk_softmax_loss_impl


def test_uniform_unit_directions_exclude_self():
    x=torch.tensor([[[0.,0.],[3.,0.],[3.,4.]]])
    actual=uniform_toward(x,x,exclude_self=True)
    expected=torch.zeros_like(x)
    for i in range(3):
        expected[0,i]=sum((x[0,j]-x[0,i])/torch.linalg.vector_norm(x[0,j]-x[0,i]) for j in range(3) if j!=i)/2
    torch.testing.assert_close(actual,expected)


def test_uniform_matches_reference_when_only_finite_softmax_logits_are_zeroed(monkeypatch):
    torch.manual_seed(19)
    pred=torch.randn(2,6,3,requires_grad=True)
    target=torch.randn(2,3,requires_grad=True)
    tau=[102.4,25.6,10.24]
    new,_=uniform_riesz_loss(pred,target,tau)
    new_grad=torch.autograd.grad(new.mean(),pred,retain_graph=True)[0]
    # Reference source is unchanged; override logits only inside this test.
    original=torch.softmax
    def zero_logits(value,dim):
        logits=torch.where(torch.isneginf(value),value,torch.zeros_like(value))
        return original(logits,dim=dim)
    monkeypatch.setattr(torch,'softmax',zero_logits)
    ref,_=_riesz_topk_softmax_loss_impl(pred,target[:,None],tau_list=tau,
        normalize_per_temperature=True,epsilon=1e-6,direction_epsilon=1e-8)
    torch.testing.assert_close(new,ref,rtol=2e-5,atol=2e-5)
    ref_grad=torch.autograd.grad(ref.mean(),pred,retain_graph=True)[0]
    torch.testing.assert_close(new_grad,ref_grad,rtol=2e-5,atol=2e-5)
    new.mean().backward()
    assert target.grad is None


def test_uniform_has_finite_zero_field_and_permutation_equivariance():
    x=torch.zeros(2,4,3,requires_grad=True); y=torch.zeros(2,3)
    loss,_=uniform_riesz_loss(x,y,[102.4,25.6,10.24])
    loss.mean().backward()
    assert torch.isfinite(loss).all() and torch.isfinite(x.grad).all() and loss.sum()==0
    torch.manual_seed(20); x=torch.randn(2,4,3); y=torch.randn(2,3)
    a,_=uniform_riesz_loss(x,y,[102.4,25.6,10.24])
    b,_=uniform_riesz_loss(x[:,[2,0,3,1]],y,[102.4,25.6,10.24])
    torch.testing.assert_close(a,b)
