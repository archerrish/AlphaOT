"""One global TV-unbalanced sub-coupling LP, with explicit missing-edge masks."""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def unbalanced_transport(a,b,cost,penalty=.35,available=None):
    a,b,cost=np.asarray(a,float),np.asarray(b,float),np.asarray(cost,float)
    if a.ndim!=1 or b.ndim!=1 or not len(a) or not len(b) or cost.shape!=(len(a),len(b)):
        raise ValueError('Invalid transport dimensions')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or (a<0).any() or (b<0).any():
        raise ValueError('Invalid capacities')
    if not np.isfinite(penalty) or penalty<=0:raise ValueError('Invalid lambda')
    mask=np.ones(cost.shape,bool) if available is None else np.asarray(available,bool)
    if mask.shape!=cost.shape or not np.isfinite(cost[mask]).all() or (cost[mask]<0).any():
        raise ValueError('Invalid observed costs')
    eligible=mask & (cost<2*penalty) & (a[:,None]>0) & (b[None,:]>0)
    i,j=np.where(eligible);n=len(i)
    # A fixed zero dummy variable ensures exactly one LP even with no eligible edges.
    columns=np.arange(n)
    constraints=coo_matrix((np.ones(2*n),(np.r_[i,len(a)+j],np.r_[columns,columns])),shape=(len(a)+len(b),max(n,1))).tocsr()
    fit=linprog(cost[i,j]-2*penalty if n else [0.],A_ub=constraints,b_ub=np.r_[a,b],
                bounds=(0,None) if n else (0,0),method='highs')
    if not fit.success:raise RuntimeError(f'HiGHS failed: {fit.message}')
    plan=np.zeros(cost.shape);plan[i,j]=fit.x if n else []
    ua,ub=a-plan.sum(axis=1),b-plan.sum(axis=0)
    if min(ua.min(),ub.min()) < -1e-7:raise RuntimeError('Global capacity exceeded')
    objective=float(np.dot(plan[mask],cost[mask])+penalty*(ua.sum()+ub.sum()))
    return dict(plan=plan,transported_mass=float(plan.sum()),objective=objective,
                gain_over_unmatched=float(penalty*(a.sum()+b.sum())-objective),
                unmatched_a=np.maximum(ua,0),unmatched_b=np.maximum(ub,0),lp_count=1,
                eligible=eligible)
