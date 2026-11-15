#include <metal_stdlib>
using namespace metal;

// Each seed has private optimizer state. Geometry and matching are parallel
// over primitives/observations; the small 8x8 optimizer update stays on device.
constant float DEG = 0.017453292519943295f;
constant float XS[8] = {.25f,.25f,.25f,8.f,8.f,25.f,15.f,15.f};
constant float SIG[8] = {.6f,.6f,.6f,25.f,25.f,70.f,180.f,180.f};
inline float3 rotate_axis(float3 p, uint axis, float degrees) {
    float a=degrees*DEG, c=cos(a), s=sin(a);
    if(axis==0) return float3(p.x,c*p.y-s*p.z,s*p.y+c*p.z);
    if(axis==1) return float3(c*p.x+s*p.z,p.y,-s*p.x+c*p.z);
    return float3(c*p.x-s*p.y,s*p.x+c*p.y,p.z);
}
inline float3 unit(uint axis) { return axis==0?float3(1,0,0):(axis==1?float3(0,1,0):float3(0,0,1)); }
inline float3x3 skew3(float3 v) { return float3x3(float3(0,v.z,-v.y),float3(-v.z,0,v.x),float3(v.y,-v.x,0)); }
inline void rodrigues(float3 v, thread float3x3& r, thread float3x3& right) {
    float q=dot(v,v), theta=sqrt(max(q,1e-16f));
    float a,b,c;
    if(q<1e-8f) { a=1-q/6+q*q/120; b=.5f-q/24+q*q/720; c=1.f/6-q/120+q*q/5040; }
    else { a=sin(theta)/theta; b=(1-cos(theta))/q; c=(theta-sin(theta))/(q*theta); }
    float3x3 k=skew3(v), k2=k*k, eye=float3x3(1.f);
    r=eye+a*k+b*k2; right=eye-b*k+c*k2;
}
inline bool clear_slab(float3 camera, float3 endpoint, uint axis, uint layer, float half_size) {
    float3 low(-half_size),high(half_size);
    low[axis]=-half_size+float(layer)*(2*half_size/3);
    high[axis]=-half_size+float(layer+1)*(2*half_size/3);
    float3 ray=endpoint-camera;
    float entry=0,exit=INFINITY;
    bool inside=true;
    for(uint j=0;j<3;j++) {
        if(abs(ray[j])<1e-12f) { inside=inside && camera[j]>=low[j]-1e-7f && camera[j]<=high[j]+1e-7f; }
        else {
            float a=(low[j]-camera[j])/ray[j],b=(high[j]-camera[j])/ray[j];
            entry=max(entry,min(a,b)); exit=min(exit,max(a,b));
        }
    }
    return !(inside && entry<=exit+1e-7f) || entry>=1-1e-5f;
}
inline float3 load3(device const float* p) { return float3(p[0],p[1],p[2]); }
inline void project(float3 p, float3x3 r, float3x3 right, float3 t, uint axis,
                    constant float* calibration, thread float2& pixel, thread float* jac, thread float2& angle_jac) {
    float3 xyz=r*p+t;
    float x=xyz.x/xyz.z,y=xyz.y/xyz.z;
    float u=x,v=y,a=1,b=0,c=0,d=1;
    // fx,fy,cx,cy, followed by OpenCV's twelve non-tilt coefficients.
    float k1=calibration[4],k2=calibration[5],p1=calibration[6],p2=calibration[7],k3=calibration[8];
    float k4=calibration[9],k5=calibration[10],k6=calibration[11];
    float s1=calibration[12],s2=calibration[13],s3=calibration[14],s4=calibration[15];
    float r2=x*x+y*y,r4=r2*r2,r6=r4*r2;
    float num=1+k1*r2+k2*r4+k3*r6,den=1+k4*r2+k5*r4+k6*r6;
    float radial=num/den,derivative=((k1+2*k2*r2+3*k3*r4)*den-num*(k4+2*k5*r2+3*k6*r4))/(den*den);
    u=x*radial+2*p1*x*y+p2*(r2+2*x*x)+s1*r2+s2*r4;
    v=y*radial+p1*(r2+2*y*y)+2*p2*x*y+s3*r2+s4*r4;
    a=radial+2*x*x*derivative+2*p1*y+6*p2*x+2*s1*x+4*s2*r2*x;
    b=2*x*y*derivative+2*p1*x+2*p2*y+2*s1*y+4*s2*r2*y;
    c=2*x*y*derivative+2*p1*x+2*p2*y+2*s3*x+4*s4*r2*x;
    d=radial+2*y*y*derivative+6*p1*y+2*p2*x+2*s3*y+4*s4*r2*y;
    pixel=float2(calibration[0]*u+calibration[2],calibration[1]*v+calibration[3]);
    float3 grad[2]={calibration[0]*float3(a,b,-a*x-b*y)/xyz.z,
                    calibration[1]*float3(c,d,-c*x-d*y)/xyz.z};
    float3 angle=r*cross(unit(axis),p)*DEG;
    for(uint row=0;row<2;row++) {
        for(uint j=0;j<3;j++) { jac[row*6+j]=dot(grad[row],-(r*cross(p,right[j]))); jac[row*6+3+j]=grad[row][j]; }
        angle_jac[row]=dot(grad[row],angle);
    }
}

// config: batch,point count,face count,landmark count,sticker count.
// Primitive result: projected center(2), residual Jacobian(16), visible(1).
kernel void geometry_project(device const float* parameters, device const float* objects,
    device const int* ids, device const int* axes, constant float* calibration,
    constant int* config, device float* output, uint index [[thread_position_in_grid]]) {
    uint P=config[1],F=config[2],Q=P+F;
    if(index>=uint(config[0])*Q) return;
    uint seed=index/Q,primitive=index%Q,axis=axes[seed];
    device const float* x=parameters+seed*8;
    float3x3 r,right; rodrigues(load3(x),r,right);
    float3 t=load3(x+3),camera=-(transpose(r)*t);
    device const float* base=objects+index*30;
    uint layer=ids[index*2];
    float angle=layer==0?x[6]:(layer==2?x[7]:0);
    uint corners=primitive<P?1:4;
    float2 center(0),angle_jac(0);float jac[12];for(uint j=0;j<12;j++)jac[j]=0;
    float3 base_center(0);
    for(uint corner=0;corner<corners;corner++) {
        float3 p0=load3(base+corner*3),p=rotate_axis(p0,axis,angle);
        float2 uv,aj;float local_jac[12];project(p,r,right,t,axis,calibration,uv,local_jac,aj);
        center+=uv;angle_jac+=aj;for(uint j=0;j<12;j++)jac[j]+=local_jac[j];base_center+=p0;
    }
    center/=float(corners);angle_jac/=float(corners);base_center/=float(corners);
    float3 object_center=rotate_axis(base_center,axis,angle);
    bool visible=false;
    for(uint j=0;j<(primitive<P?6:1);j++) visible=visible || dot(rotate_axis(load3(base+12+j*3),axis,angle),camera-object_center)>0;
    for(uint blocker=0;blocker<3;blocker++) {
        float blocker_angle=blocker==0?x[6]:(blocker==2?x[7]:0);
        visible=visible && clear_slab(rotate_axis(camera,axis,-blocker_angle),
            rotate_axis(base_center,axis,angle-blocker_angle),axis,blocker,calibration[16]);
    }
    device float* dest=output+index*19;
    dest[0]=center.x;dest[1]=center.y;
    for(uint row=0;row<2;row++) {
        for(uint j=0;j<6;j++)dest[2+row*8+j]=jac[row*6+j]/float(corners);
        dest[2+row*8+6]=layer==0?angle_jac[row]:0;
        dest[2+row*8+7]=layer==2?angle_jac[row]:0;
    }
    dest[18]=visible?1.f:0.f;
}

kernel void match_residuals(device const float* projections, device const int* ids,
    device const float* observations, device const int* classes, constant int* config,
    constant float& scale, device float* output, uint index [[thread_position_in_grid]]) {
    uint P=config[1],F=config[2],N=config[3],M=config[4],O=N+M,Q=P+F;
    if(index>=uint(config[0])*O)return;
    uint seed=index/O,obs=index%O;
    bool sticker=obs>=N;
    uint first=sticker?P:0,last=sticker?Q:P,best=first;
    float best_distance=INFINITY;
    float2 xy=float2(observations[obs*2],observations[obs*2+1]);
    for(uint j=first;j<last;j++) {
        uint k=seed*Q+j;
        if(projections[k*19+18]<.5f)continue;
        if(!sticker && classes[obs]>=0 && ids[k*2+1]>=0 && classes[obs]!=ids[k*2+1])continue;
        float2 uv=float2(projections[k*19],projections[k*19+1]);
        float dist=dot(uv-xy,uv-xy);
        if(dist<best_distance){best_distance=dist;best=j;}
    }
    device float* dest=output+index*18;
    device const float* source=projections+(seed*Q+best)*19;
    bool matched=isfinite(best_distance);
    for(uint row=0;row<2;row++) {
        dest[row]=matched?(source[row]-xy[row])/scale:40.f;
        for(uint j=0;j<8;j++)dest[2+row*8+j]=matched?source[2+row*8+j]/scale:0.f;
    }
}

// State: accepted x(8), cost, damping, gradient(8), Hessian(64), initial cost.
kernel void optimizer_step(device float* parameters, device const float* seeds,
    device const float* priors, device const float* residuals, constant int* config,
    constant int& iteration, constant float& min_depth, device float* states,
    uint seed [[thread_position_in_grid]]) {
    if(seed>=uint(config[0]))return;
    uint O=config[3]+config[4];
    device float* x=parameters+seed*8;
    device const float* start=seeds+seed*8;
    device const float* prior=priors+seed*8;
    device float* state=states+seed*83;
    float H[64],g[8],cost=0;
    bool free[8];for(uint j=0;j<8;j++){g[j]=0;free[j]=j<6 || start[j]!=0;}
    for(uint j=0;j<64;j++)H[j]=0;
    for(uint row=0;row<2*O+8;row++) {
        float residual,jac[8];
        if(row<2*O) {
            uint obs=row/2,coordinate=row%2;
            device const float* data=residuals+(seed*O+obs)*18;
            residual=data[coordinate];
            for(uint j=0;j<8;j++)jac[j]=free[j]?data[2+coordinate*8+j]*XS[j]:0;
        } else {
            uint regularizer=row-2*O;
            residual=(x[regularizer]-prior[regularizer])/SIG[regularizer];
            for(uint j=0;j<8;j++)jac[j]=j==regularizer && free[j]?XS[j]/SIG[j]:0;
        }
        float root=sqrt(1+residual*residual/9);
        // Equivalent soft_l1 cost without cancellation near zero.
        cost+=residual*residual/(root+1);
        for(uint j=0;j<8;j++) {
            g[j]+=jac[j]*residual/root;
            for(uint k=0;k<=j;k++)H[j*8+k]+=jac[j]*jac[k]/(root*root*root);
        }
    }
    for(uint j=0;j<8;j++)for(uint k=j+1;k<8;k++)H[j*8+k]=H[k*8+j];
    float damping=iteration==0?.01f:state[9];
    bool accepted=iteration==0 || (isfinite(cost) && cost<state[8]);
    if(iteration==0)state[82]=cost;
    else {
        float predicted=0,linear=0,quadratic=0,step[8];
        for(uint j=0;j<8;j++) {step[j]=(x[j]-state[j])/XS[j];linear+=state[10+j]*step[j];}
        for(uint j=0;j<8;j++)for(uint k=0;k<8;k++)quadratic+=step[j]*state[18+j*8+k]*step[k];
        predicted=-linear-.5f*quadratic;
        float ratio=(state[8]-cost)/max(predicted,1e-12f);
        if(accepted && ratio>.75f)damping/=3;
        else if(!accepted || ratio<.25f)damping*=4;
    }
    damping=clamp(damping,1e-8f,1e8f);state[9]=damping;
    if(accepted) {
        for(uint j=0;j<8;j++){state[j]=x[j];state[10+j]=g[j];}
        for(uint j=0;j<64;j++)state[18+j]=H[j];state[8]=cost;
    }
    float L[64],delta[8],y[8];for(uint j=0;j<64;j++)L[j]=0;
    for(uint j=0;j<8;j++) {
        for(uint k=0;k<=j;k++) {
            float a=state[18+j*8+k];
            if(j==k)a+=damping*max(state[18+j*8+j],1e-8f)+(free[j]?0.f:1.f);
            for(uint z=0;z<k;z++)a-=L[j*8+z]*L[k*8+z];
            L[j*8+k]=j==k?sqrt(max(a,1e-12f)):a/L[k*8+k];
        }
        float a=-state[10+j];for(uint k=0;k<j;k++)a-=L[j*8+k]*y[k];y[j]=a/L[j*8+j];
    }
    for(int j=7;j>=0;j--) {float a=y[j];for(uint k=j+1;k<8;k++)a-=L[k*8+j]*delta[k];delta[j]=a/L[j*8+j];}
    for(uint j=0;j<8;j++) {
        float radius=j<3?.8f:(j<5?40.f:100.f);
        float lo=j<6?start[j]-radius:-45.f,hi=j<6?start[j]+radius:45.f;
        if(j==5)lo=max(lo,min_depth);
        x[j]=free[j]?clamp(state[j]+delta[j]*XS[j],lo,hi):start[j];
    }
}
