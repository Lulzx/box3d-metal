#include <metal_stdlib>
using namespace metal;
#include "metal_abi.h"
V3W add3(V3W a, V3W b) { return V3W{a.X+b.X,a.Y+b.Y,a.Z+b.Z}; }
V3W sub3(V3W a, V3W b) { return V3W{a.X-b.X,a.Y-b.Y,a.Z-b.Z}; }
V3W mul3(float4 s, V3W a) { return V3W{s*a.X,s*a.Y,s*a.Z}; }
V3W cross3(V3W a, V3W b) {
  return V3W{a.Y*b.Z-a.Z*b.Y,a.Z*b.X-a.X*b.Z,a.X*b.Y-a.Y*b.X};
}
float4 dot3(V3W a, V3W b) { return a.X*b.X+a.Y*b.Y+a.Z*b.Z; }
V2W mul_sym2(Sym2W m, V2W a) { return V2W{m.cxx*a.x+m.cxy*a.y,m.cxy*a.x+m.cyy*a.y}; }
V3W mul_sym3(Sym3W m, V3W a) {
  return V3W{m.cxx*a.X+m.cxy*a.Y+m.cxz*a.Z,
             m.cxy*a.X+m.cyy*a.Y+m.cyz*a.Z,
             m.cxz*a.X+m.cyz*a.Y+m.czz*a.Z};
}
V3W rotate3(QW q, V3W a) {
  V3W t1=cross3(q.V,a); V3W t2=add3(t1,mul3(q.S,a));
  return add3(a,mul3(float4(2.0f),cross3(q.V,t2)));
}
BodyW gather_bodies(device BodyState* states, const device int* indices) {
  BodyW b; b.v=V3W{float4(0.0f),float4(0.0f),float4(0.0f)};
  b.w=V3W{float4(0.0f),float4(0.0f),float4(0.0f)};
  b.dp=V3W{float4(0.0f),float4(0.0f),float4(0.0f)};
  b.dq.V=V3W{float4(0.0f),float4(0.0f),float4(0.0f)}; b.dq.S=float4(1.0f);
  for (uint lane=0; lane<4; ++lane) { int index=indices[lane]-1; if (index < 0) continue;
    BodyState s=states[index]; b.v.X[lane]=s.lvx; b.v.Y[lane]=s.lvy; b.v.Z[lane]=s.lvz;
    b.w.X[lane]=s.avx; b.w.Y[lane]=s.avy; b.w.Z[lane]=s.avz;
    b.dp.X[lane]=s.dpx; b.dp.Y[lane]=s.dpy; b.dp.Z[lane]=s.dpz;
    b.dq.V.X[lane]=s.qx; b.dq.V.Y[lane]=s.qy; b.dq.V.Z[lane]=s.qz; b.dq.S[lane]=s.qw;
  } return b;
}
void scatter_bodies(device BodyState* states, const device int* indices, BodyW b) {
  for (uint lane=0; lane<4; ++lane) { int index=indices[lane]-1; if (index < 0) continue;
    BodyState s=states[index]; if ((s.flags & 0x00001000u)==0u) continue;
    float3 v=float3(b.v.X[lane],b.v.Y[lane],b.v.Z[lane]);
    float3 w=float3(b.w.X[lane],b.w.Y[lane],b.w.Z[lane]);
    if (s.flags & 0x1u) v.x=0.0f; if (s.flags & 0x2u) v.y=0.0f; if (s.flags & 0x4u) v.z=0.0f;
    if (s.flags & 0x8u) w.x=0.0f; if (s.flags & 0x10u) w.y=0.0f; if (s.flags & 0x20u) w.z=0.0f;
    s.lvx=v.x; s.lvy=v.y; s.lvz=v.z; s.avx=w.x; s.avy=w.y; s.avz=w.z; states[index]=s;
  }
}
S3 sadd(S3 a,S3 b){return S3{a.x+b.x,a.y+b.y,a.z+b.z};} S3 ssub(S3 a,S3 b){return S3{a.x-b.x,a.y-b.y,a.z-b.z};}
S3 smul(float s,S3 a){return S3{s*a.x,s*a.y,s*a.z};} float sdot(S3 a,S3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
S3 scross(S3 a,S3 b){return S3{a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
S3 smulm(SM3 m,S3 a){return S3{m.cx.x*a.x+m.cy.x*a.y+m.cz.x*a.z,m.cx.y*a.x+m.cy.y*a.y+m.cz.y*a.z,m.cx.z*a.x+m.cy.z*a.y+m.cz.z*a.z};}
S3 sperp(S3 a){S3 p=(a.x < -0.5f || a.x > 0.5f)?S3{a.y,-a.x,0.0f}:S3{0.0f,a.z,-a.y};return smul(rsqrt(sdot(p,p)),p);}
SM3 sinvert3(SM3 m){float det=sdot(m.cx,scross(m.cy,m.cz));if(fabs(det)<=1.17549435e-35f)return SM3{S3{0,0,0},S3{0,0,0},S3{0,0,0}};
  float inv=1.0f/det;S3 a=smul(inv,scross(m.cy,m.cz)),b=smul(inv,scross(m.cz,m.cx)),c=smul(inv,scross(m.cx,m.cy));
  return SM3{S3{a.x,b.x,c.x},S3{a.y,b.y,c.y},S3{a.z,b.z,c.z}};}
SM3 load_inertia(const device BodyProperties* p,int index){if(index<0)return SM3{S3{0,0,0},S3{0,0,0},S3{0,0,0}};
  const device float* v=p[index].invInertiaWorld;return SM3{S3{v[0],v[1],v[2]},S3{v[3],v[4],v[5]},S3{v[6],v[7],v[8]}};}
S3 srotate(S3 qv,float qs,S3 a){S3 t1=scross(qv,a);S3 t2=sadd(t1,smul(qs,a));return sadd(a,smul(2.0f,scross(qv,t2)));}
BodyS load_body(device BodyState* states,int index){BodyS b; b.v=S3{0,0,0};b.w=S3{0,0,0};b.dp=S3{0,0,0};b.dqv=S3{0,0,0};b.dqs=1;
  if(index>=0){BodyState s=states[index];b.v=S3{s.lvx,s.lvy,s.lvz};b.w=S3{s.avx,s.avy,s.avz};b.dp=S3{s.dpx,s.dpy,s.dpz};b.dqv=S3{s.qx,s.qy,s.qz};b.dqs=s.qw;}return b;}
void store_body(device BodyState* states,int index,BodyS b){if(index<0)return;BodyState s=states[index];if((s.flags&0x1000u)==0u)return;
  if(s.flags&1u)b.v.x=0;if(s.flags&2u)b.v.y=0;if(s.flags&4u)b.v.z=0;if(s.flags&8u)b.w.x=0;if(s.flags&16u)b.w.y=0;if(s.flags&32u)b.w.z=0;
  s.lvx=b.v.x;s.lvy=b.v.y;s.lvz=b.v.z;s.avx=b.w.x;s.avy=b.w.y;s.avz=b.w.z;states[index]=s;}
kernel void b3_prepare_contacts(const device uint* indices [[buffer(0)]],const device PrepareInput* inputs [[buffer(1)]],
  const device ConvexManifoldResult* table [[buffer(2)]],const device BodyProperties* properties [[buffer(3)]],
  const device BodyState* states [[buffer(4)]],device ContactWide* constraints [[buffer(5)]],
  device atomic_uint* status [[buffer(6)]],constant PrepareParams& p [[buffer(7)]],uint tid [[thread_position_in_grid]]){
  if(tid>=p.wideCount)return;device ContactWide& c=constraints[tid];
  for(uint lane=0;lane<4u;++lane){uint contactId=indices[4u*tid+lane];if(contactId==0xffffffffu)continue;
    if(contactId>=p.tableCount){atomic_fetch_or_explicit(status,1u,memory_order_relaxed);continue;}PrepareInput in=inputs[contactId];
    ConvexManifoldResult mr=table[contactId];if(in.contactId!=contactId||in.generation!=p.generation||mr.eligible==0u||mr.touching==0u||mr.contactId!=contactId||
      mr.inputIndex!=contactId||mr.pointCount==0u||mr.pointCount>4u){atomic_fetch_or_explicit(status,2u,memory_order_relaxed);continue;}
    uint pointCount=mr.pointCount;int ia=in.indexA,ib=in.indexB;c.indexA[lane]=ia+1;c.indexB[lane]=ib+1;c.pointCounts[lane]=int(pointCount);c.manifolds[lane]=in.manifold;
    float ma=ia>=0?properties[ia].invMass:0.0f,mb=ib>=0?properties[ib].invMass:0.0f;c.invMassA[lane]=ma;c.invMassB[lane]=mb;
    SM3 iA=load_inertia(properties,ia),iB=load_inertia(properties,ib);
    c.invIA.cxx[lane]=iA.cx.x;c.invIA.cxy[lane]=iA.cy.x;c.invIA.cxz[lane]=iA.cz.x;c.invIA.cyy[lane]=iA.cy.y;c.invIA.cyz[lane]=iA.cz.y;c.invIA.czz[lane]=iA.cz.z;
    c.invIB.cxx[lane]=iB.cx.x;c.invIB.cxy[lane]=iB.cy.x;c.invIB.cxz[lane]=iB.cz.x;c.invIB.cyy[lane]=iB.cy.y;c.invIB.cyz[lane]=iB.cz.y;c.invIB.czz[lane]=iB.cz.z;
    S3 n=S3{mr.nx,mr.ny,mr.nz},t1=sperp(n),t2=scross(t1,n),tv=S3{in.tangentVelocityX,in.tangentVelocityY,in.tangentVelocityZ};
    c.normal.X[lane]=n.x;c.normal.Y[lane]=n.y;c.normal.Z[lane]=n.z;c.tangent1.X[lane]=t1.x;c.tangent1.Y[lane]=t1.y;c.tangent1.Z[lane]=t1.z;
    c.tangent2.X[lane]=t2.x;c.tangent2.Y[lane]=t2.y;c.tangent2.Z[lane]=t2.z;c.friction[lane]=in.friction;c.restitution[lane]=in.restitution;
    c.rollingResistance[lane]=in.rollingResistance;c.tangentVelocity1[lane]=sdot(tv,t1);c.tangentVelocity2[lane]=sdot(tv,t2);
    Softness soft=(ia<0||ib<0)?p.staticSoftness:p.contactSoftness;c.biasRate[lane]=soft.biasRate;c.massScale[lane]=soft.massScale;c.impulseScale[lane]=soft.impulseScale;
    BodyState sa={};BodyState sb={};if(ia>=0)sa=states[ia];if(ib>=0)sb=states[ib];S3 va=S3{sa.lvx,sa.lvy,sa.lvz},wa=S3{sa.avx,sa.avy,sa.avz};
    S3 vb=S3{sb.lvx,sb.lvy,sb.lvz},wb=S3{sb.avx,sb.avy,sb.avz},centerA=S3{0,0,0},centerB=S3{0,0,0};float totalWeight=0.0f;
    for(uint j=0;j<pointCount;++j){PreparePoint mp=in.points[j];device PointWide& cp=c.points[j];S3 rA=S3{mp.anchorAX,mp.anchorAY,mp.anchorAZ},rB=S3{mp.anchorBX,mp.anchorBY,mp.anchorBZ};
      float weight=clamp(2.0f-mp.separation*p.invTau,1.0e-10f,1.0f);centerA=sadd(centerA,smul(weight,rA));centerB=sadd(centerB,smul(weight,rB));totalWeight+=weight;
      cp.anchorAs.X[lane]=rA.x;cp.anchorAs.Y[lane]=rA.y;cp.anchorAs.Z[lane]=rA.z;cp.anchorBs.X[lane]=rB.x;cp.anchorBs.Y[lane]=rB.y;cp.anchorBs.Z[lane]=rB.z;
      cp.baseSeparations[lane]=mp.separation-sdot(ssub(rB,rA),n);cp.normalImpulses[lane]=p.warmStartScale*mp.normalImpulse;cp.totalNormalImpulses[lane]=0.0f;
      S3 rnA=scross(rA,n),rnB=scross(rB,n);float k=ma+mb+sdot(rnA,smulm(iA,rnA))+sdot(rnB,smulm(iB,rnB));cp.normalMasses[lane]=k>0.0f?1.0f/k:0.0f;
      S3 vrA=sadd(va,scross(wa,rA)),vrB=sadd(vb,scross(wb,rB));cp.relativeVelocities[lane]=sdot(n,ssub(vrB,vrA));}
    float iw=1.0f/totalWeight;centerA=smul(iw,centerA);centerB=smul(iw,centerB);c.centerA.X[lane]=centerA.x;c.centerA.Y[lane]=centerA.y;c.centerA.Z[lane]=centerA.z;
    c.centerB.X[lane]=centerB.x;c.centerB.Y[lane]=centerB.y;c.centerB.Z[lane]=centerB.z;
    for(uint j=0;j<pointCount;++j){S3 rA=S3{in.points[j].anchorAX,in.points[j].anchorAY,in.points[j].anchorAZ};c.points[j].leverArms[lane]=length(float3(rA.x-centerA.x,rA.y-centerA.y,rA.z-centerA.z));}
    for(uint j=pointCount;j<4u;++j){device PointWide& cp=c.points[j];cp.anchorAs.X[lane]=0;cp.anchorAs.Y[lane]=0;cp.anchorAs.Z[lane]=0;cp.anchorBs.X[lane]=0;cp.anchorBs.Y[lane]=0;cp.anchorBs.Z[lane]=0;
      cp.baseSeparations[lane]=0;cp.normalImpulses[lane]=0;cp.totalNormalImpulses[lane]=0;cp.normalMasses[lane]=0;cp.leverArms[lane]=0;cp.relativeVelocities[lane]=0;}
    S3 ra1=scross(centerA,t1),ra2=scross(centerA,t2),rb1=scross(centerB,t1),rb2=scross(centerB,t2);float kxx=ma+mb+sdot(ra1,smulm(iA,ra1))+sdot(rb1,smulm(iB,rb1));
    float kyy=ma+mb+sdot(ra2,smulm(iA,ra2))+sdot(rb2,smulm(iB,rb2)),kxy=sdot(ra1,smulm(iA,ra2))+sdot(rb1,smulm(iB,rb2));float det=kxx*kyy-kxy*kxy;
    float mxx=0,mxy=0,myy=0;if(fabs(det)>1.17549435e-35f){float id=1.0f/det;mxx=id*kyy;mxy=-id*kxy;myy=id*kxx;}c.tangentMass.cxx[lane]=mxx;c.tangentMass.cxy[lane]=mxy;c.tangentMass.cyy[lane]=myy;
    S3 fi=S3{in.frictionImpulseX,in.frictionImpulseY,in.frictionImpulseZ};c.frictionImpulse.x[lane]=p.warmStartScale*sdot(fi,t1);c.frictionImpulse.y[lane]=p.warmStartScale*sdot(fi,t2);
    SM3 sum=SM3{sadd(iA.cx,iB.cx),sadd(iA.cy,iB.cy),sadd(iA.cz,iB.cz)};float kt=sdot(n,smulm(sum,n));c.twistMass[lane]=kt>0.0f?1.0f/kt:0.0f;c.twistImpulse[lane]=p.warmStartScale*in.twistImpulse;
    SM3 rm=sinvert3(sum);c.rollingMass.cxx[lane]=rm.cx.x;c.rollingMass.cxy[lane]=rm.cy.x;c.rollingMass.cxz[lane]=rm.cz.x;c.rollingMass.cyy[lane]=rm.cy.y;c.rollingMass.cyz[lane]=rm.cz.y;c.rollingMass.czz[lane]=rm.cz.z;
    c.rollingImpulse.X[lane]=p.warmStartScale*in.rollingImpulseX;c.rollingImpulse.Y[lane]=p.warmStartScale*in.rollingImpulseY;c.rollingImpulse.Z[lane]=p.warmStartScale*in.rollingImpulseZ;
  }}
kernel void b3_warm_start_contacts(device BodyState* states [[buffer(0)]],
                                 device ContactWide* constraints [[buffer(1)]],
                                 constant ContactParams& p [[buffer(2)]],
                                 uint tid [[thread_position_in_grid]]) {
  if (tid >= p.count) return; device ContactWide& c=constraints[p.offset+tid];
  BodyW a=gather_bodies(states,c.indexA); BodyW b=gather_bodies(states,c.indexB);
  int pointCount=max(max(c.pointCounts[0],c.pointCounts[1]),max(c.pointCounts[2],c.pointCounts[3]));
  for (int j=0; j<pointCount; ++j) { device PointWide& cp=c.points[j];
    V3W impulse=mul3(cp.normalImpulses,c.normal);
    a.w=sub3(a.w,mul_sym3(c.invIA,cross3(cp.anchorAs,impulse))); a.v=sub3(a.v,mul3(c.invMassA,impulse));
    b.w=add3(b.w,mul_sym3(c.invIB,cross3(cp.anchorBs,impulse))); b.v=add3(b.v,mul3(c.invMassB,impulse));
  }
  V3W frictionImpulse=add3(mul3(c.frictionImpulse.x,c.tangent1),mul3(c.frictionImpulse.y,c.tangent2));
  a.w=sub3(a.w,mul_sym3(c.invIA,cross3(c.centerA,frictionImpulse))); a.v=sub3(a.v,mul3(c.invMassA,frictionImpulse));
  b.w=add3(b.w,mul_sym3(c.invIB,cross3(c.centerB,frictionImpulse))); b.v=add3(b.v,mul3(c.invMassB,frictionImpulse));
  V3W twist=mul3(c.twistImpulse,c.normal); a.w=sub3(a.w,mul_sym3(c.invIA,twist)); b.w=add3(b.w,mul_sym3(c.invIB,twist));
  a.w=sub3(a.w,mul_sym3(c.invIA,c.rollingImpulse)); b.w=add3(b.w,mul_sym3(c.invIB,c.rollingImpulse));
  scatter_bodies(states,c.indexA,a); scatter_bodies(states,c.indexB,b);
}
kernel void b3_solve_contacts(device BodyState* states [[buffer(0)]],
                            device ContactWide* constraints [[buffer(1)]],
                            constant ContactParams& p [[buffer(2)]],
                            uint tid [[thread_position_in_grid]]) {
  if (tid >= p.count) return; device ContactWide& c=constraints[p.offset+tid];
  BodyW a=gather_bodies(states,c.indexA); BodyW b=gather_bodies(states,c.indexB);
  int pointCount=max(max(c.pointCounts[0],c.pointCounts[1]),max(c.pointCounts[2],c.pointCounts[3]));
  float4 biasRate=p.useBias!=0u ? c.massScale*c.biasRate : float4(0.0f);
  float4 massScale=p.useBias!=0u ? c.massScale : float4(1.0f);
  float4 impulseScale=p.useBias!=0u ? c.impulseScale : float4(0.0f);
  V3W dp=sub3(b.dp,a.dp); float4 totalNormal=float4(0.0f); float4 totalTwist=float4(0.0f);
  for (int j=0; j<pointCount; ++j) { device PointWide& cp=c.points[j];
    V3W rsA=rotate3(a.dq,cp.anchorAs); V3W rsB=rotate3(b.dq,cp.anchorBs);
    float4 separation=dot3(c.normal,add3(dp,sub3(rsB,rsA)))+cp.baseSeparations;
    bool4 speculative=separation>float4(0.0f);
    float4 bias=select(max(biasRate*separation,float4(p.contactSpeed)),separation*float4(p.invH),speculative);
    float4 pointMassScale=select(massScale,float4(1.0f),speculative);
    float4 pointImpulseScale=select(impulseScale,float4(0.0f),speculative);
    V3W vrA=add3(a.v,cross3(a.w,cp.anchorAs)); V3W vrB=add3(b.v,cross3(b.w,cp.anchorBs));
    float4 vn=dot3(sub3(vrB,vrA),c.normal);
    float4 neg=cp.normalMasses*(pointMassScale*vn+bias)+pointImpulseScale*cp.normalImpulses;
    float4 next=max(cp.normalImpulses-neg,float4(0.0f)); float4 delta=next-cp.normalImpulses;
    cp.normalImpulses=next; cp.totalNormalImpulses+=next; totalNormal+=next; totalTwist+=cp.leverArms*next; V3W impulse=mul3(delta,c.normal);
    a.w=sub3(a.w,mul_sym3(c.invIA,cross3(cp.anchorAs,impulse))); a.v=sub3(a.v,mul3(c.invMassA,impulse));
    b.w=add3(b.w,mul_sym3(c.invIB,cross3(cp.anchorBs,impulse))); b.v=add3(b.v,mul3(c.invMassB,impulse));
  }
  if (p.useBias==0u) {
    if (any(c.rollingResistance!=float4(0.0f))) {
      V3W rollingDelta=mul_sym3(c.rollingMass,sub3(a.w,b.w)); V3W oldRolling=c.rollingImpulse;
      c.rollingImpulse=add3(oldRolling,rollingDelta); float4 maxRolling=c.rollingResistance*totalNormal;
      float4 rollingLength2=dot3(c.rollingImpulse,c.rollingImpulse); bool4 clampRolling=rollingLength2>maxRolling*maxRolling+float4(1.1920929e-7f);
      float4 rollingScale=select(float4(1.0f),maxRolling/(sqrt(rollingLength2)+float4(1.1920929e-7f)),clampRolling);
      rollingScale=select(float4(0.0f),rollingScale,c.rollingResistance>float4(0.0f)); c.rollingImpulse=mul3(rollingScale,c.rollingImpulse);
      rollingDelta=sub3(c.rollingImpulse,oldRolling); a.w=sub3(a.w,mul_sym3(c.invIA,rollingDelta)); b.w=add3(b.w,mul_sym3(c.invIB,rollingDelta));
    }
    float4 twistSpeed=dot3(c.normal,sub3(b.w,a.w)); float4 maxTwist=c.friction*totalTwist; float4 oldTwist=c.twistImpulse;
    c.twistImpulse=clamp(oldTwist-c.twistMass*twistSpeed,-maxTwist,maxTwist); float4 twistDelta=c.twistImpulse-oldTwist;
    V3W angularImpulse=mul3(twistDelta,c.normal); a.w=sub3(a.w,mul_sym3(c.invIA,angularImpulse)); b.w=add3(b.w,mul_sym3(c.invIB,angularImpulse));
    V3W vrA=add3(a.v,cross3(a.w,c.centerA)); V3W vrB=add3(b.v,cross3(b.w,c.centerB)); V3W vr=sub3(vrB,vrA);
    V2W vt=V2W{dot3(vr,c.tangent1)-c.tangentVelocity1,dot3(vr,c.tangent2)-c.tangentVelocity2};
    V2W frictionDelta=mul_sym2(c.tangentMass,vt); frictionDelta.x=-frictionDelta.x; frictionDelta.y=-frictionDelta.y;
    V2W oldFriction=c.frictionImpulse; V2W nextFriction=V2W{oldFriction.x+frictionDelta.x,oldFriction.y+frictionDelta.y};
    float4 maxFriction=c.friction*totalNormal; float4 frictionLength2=nextFriction.x*nextFriction.x+nextFriction.y*nextFriction.y;
    bool4 clampFriction=frictionLength2>maxFriction*maxFriction; float4 frictionScale=select(float4(1.0f),maxFriction/(sqrt(frictionLength2)+float4(1.1920929e-7f)),clampFriction);
    nextFriction.x*=frictionScale; nextFriction.y*=frictionScale; frictionDelta=V2W{nextFriction.x-oldFriction.x,nextFriction.y-oldFriction.y}; c.frictionImpulse=nextFriction;
    V3W tangentImpulse=add3(mul3(frictionDelta.x,c.tangent1),mul3(frictionDelta.y,c.tangent2));
    a.w=sub3(a.w,mul_sym3(c.invIA,cross3(c.centerA,tangentImpulse))); a.v=sub3(a.v,mul3(c.invMassA,tangentImpulse));
    b.w=add3(b.w,mul_sym3(c.invIB,cross3(c.centerB,tangentImpulse))); b.v=add3(b.v,mul3(c.invMassB,tangentImpulse));
  } scatter_bodies(states,c.indexA,a); scatter_bodies(states,c.indexB,b);
}
kernel void b3_restitution_contacts(device BodyState* states [[buffer(0)]], device ContactWide* constraints [[buffer(1)]],
                                    constant ContactParams& p [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
  if (tid>=p.count) return; device ContactWide& c=constraints[p.offset+tid]; BodyW a=gather_bodies(states,c.indexA); BodyW b=gather_bodies(states,c.indexB);
  int pointCount=max(max(c.pointCounts[0],c.pointCounts[1]),max(c.pointCounts[2],c.pointCounts[3]));
  for (int j=0;j<pointCount;++j) { device PointWide& cp=c.points[j];
    bool4 apply=(c.restitution!=float4(0.0f))&&((cp.relativeVelocities+float4(p.restitutionThreshold))<=float4(0.0f))&&(cp.totalNormalImpulses!=float4(0.0f));
    float4 mass=select(float4(0.0f),cp.normalMasses,apply); V3W vrA=add3(a.v,cross3(a.w,cp.anchorAs)); V3W vrB=add3(b.v,cross3(b.w,cp.anchorBs));
    float4 vn=dot3(sub3(vrB,vrA),c.normal); float4 neg=mass*(vn+c.restitution*cp.relativeVelocities);
    float4 next=max(cp.normalImpulses-neg,float4(0.0f)); float4 delta=next-cp.normalImpulses; cp.normalImpulses=next; cp.totalNormalImpulses+=delta;
    V3W impulse=mul3(delta,c.normal); a.w=sub3(a.w,mul_sym3(c.invIA,cross3(cp.anchorAs,impulse))); a.v=sub3(a.v,mul3(c.invMassA,impulse));
    b.w=add3(b.w,mul_sym3(c.invIB,cross3(cp.anchorBs,impulse))); b.v=add3(b.v,mul3(c.invMassB,impulse));
  } scatter_bodies(states,c.indexA,a); scatter_bodies(states,c.indexB,b);
}
kernel void b3_store_contact_impulses(const device uint* indices [[buffer(0)]],const device ContactWide* constraints [[buffer(1)]],
  device ImpulseResult* results [[buffer(2)]],const device PrepareInput* inputs [[buffer(3)]],constant ImpulseParams& p [[buffer(4)]],uint tid [[thread_position_in_grid]]){
  if(tid>=p.wideCount)return;const device ContactWide& c=constraints[tid];
  for(uint lane=0;lane<4u;++lane){uint contactId=indices[4u*tid+lane];if(contactId==0xffffffffu||contactId>=p.tableCount)continue;
    uint pointCount=uint(c.pointCounts[lane]);if(pointCount==0u||pointCount>4u)continue;PrepareInput in=inputs[contactId];ImpulseResult r={};r.contactId=contactId;r.generation=p.generation;r.pointCount=pointCount;r.contactGeneration=in.contactGeneration;
    float f1=c.frictionImpulse.x[lane],f2=c.frictionImpulse.y[lane];r.frictionX=f1*c.tangent1.X[lane]+f2*c.tangent2.X[lane];
    r.frictionY=f1*c.tangent1.Y[lane]+f2*c.tangent2.Y[lane];r.frictionZ=f1*c.tangent1.Z[lane]+f2*c.tangent2.Z[lane];r.twistImpulse=c.twistImpulse[lane];
    r.rollingX=c.rollingImpulse.X[lane];r.rollingY=c.rollingImpulse.Y[lane];r.rollingZ=c.rollingImpulse.Z[lane];
    for(uint j=0;j<pointCount;++j){r.points[j].normalImpulse=c.points[j].normalImpulses[lane];r.points[j].totalNormalImpulse=c.points[j].totalNormalImpulses[lane];r.points[j].normalVelocity=c.points[j].relativeVelocities[lane];r.points[j].featureId=in.points[j].featureId;}
    results[contactId]=r;}
}
void warm_mesh_one(device BodyState* states,device MeshContact* contacts,device MeshManifold* manifolds,uint index){
  device MeshContact& c=contacts[index];BodyS a=load_body(states,c.indexA);BodyS b=load_body(states,c.indexB);
  for(int mi=0;mi<c.manifoldCount;++mi){device MeshManifold& m=manifolds[c.manifoldStart+mi];
    for(int j=0;j<m.pointCount;++j){device MeshPoint& cp=m.points[j];S3 impulse=smul(cp.normalImpulse,m.normal);a.w=ssub(a.w,smulm(c.invIA,scross(cp.rA,impulse)));a.v=ssub(a.v,smul(c.invMassA,impulse));b.w=sadd(b.w,smulm(c.invIB,scross(cp.rB,impulse)));b.v=sadd(b.v,smul(c.invMassB,impulse));}
    S3 f=sadd(smul(m.frictionImpulse.x,m.tangent1),smul(m.frictionImpulse.y,m.tangent2));a.w=ssub(a.w,smulm(c.invIA,scross(m.centerA,f)));a.v=ssub(a.v,smul(c.invMassA,f));b.w=sadd(b.w,smulm(c.invIB,scross(m.centerB,f)));b.v=sadd(b.v,smul(c.invMassB,f));
    S3 twist=smul(m.twistImpulse,m.normal);a.w=ssub(a.w,smulm(c.invIA,twist));b.w=sadd(b.w,smulm(c.invIB,twist));a.w=ssub(a.w,smulm(c.invIA,m.rollingImpulse));b.w=sadd(b.w,smulm(c.invIB,m.rollingImpulse));
  }store_body(states,c.indexA,a);store_body(states,c.indexB,b);}
kernel void b3_warm_start_mesh(device BodyState* s [[buffer(0)]],device MeshContact* c [[buffer(1)]],device MeshManifold* m [[buffer(2)]],constant ContactParams& p [[buffer(3)]],uint tid [[thread_position_in_grid]]){if(tid<p.count)warm_mesh_one(s,c,m,p.offset+tid);}
kernel void b3_warm_start_mesh_overflow(device BodyState* s [[buffer(0)]],device MeshContact* c [[buffer(1)]],device MeshManifold* m [[buffer(2)]],constant ContactParams& p [[buffer(3)]]){for(uint i=0;i<p.count;++i)warm_mesh_one(s,c,m,p.offset+i);}
void solve_mesh_one(device BodyState* states,device MeshContact* contacts,device MeshManifold* manifolds,constant ContactParams& p,uint index){
  device MeshContact& c=contacts[index];BodyS a=load_body(states,c.indexA);BodyS b=load_body(states,c.indexB);S3 dp=ssub(b.dp,a.dp);
  for(int mi=0;mi<c.manifoldCount;++mi){device MeshManifold& m=manifolds[c.manifoldStart+mi];float totalNormal=0,totalTwist=0;
    for(int j=0;j<m.pointCount;++j){device MeshPoint& cp=m.points[j];S3 ds=sadd(dp,ssub(srotate(b.dqv,b.dqs,cp.rB),srotate(a.dqv,a.dqs,cp.rA)));float s=sdot(ds,m.normal);s+=cp.baseSeparation;
      float bias=0,massScale=1,impulseScale=0;if(s>0)bias=s*p.invH;else if(p.useBias!=0u){bias=max(c.softness.massScale*c.softness.biasRate*s,p.contactSpeed);massScale=c.softness.massScale;impulseScale=c.softness.impulseScale;}
      S3 vrA=sadd(a.v,scross(a.w,cp.rA));S3 vrB=sadd(b.v,scross(b.w,cp.rB));float vn=sdot(ssub(vrB,vrA),m.normal);float delta=-cp.normalMass*(massScale*vn+bias)-impulseScale*cp.normalImpulse;
      float next=max(cp.normalImpulse+delta,0.0f);delta=next-cp.normalImpulse;cp.normalImpulse=next;cp.totalNormalImpulse+=next;totalNormal+=next;totalTwist+=cp.leverArm*next;
      S3 impulse=smul(delta,m.normal);a.v=ssub(a.v,smul(c.invMassA,impulse));a.w=ssub(a.w,smulm(c.invIA,scross(cp.rA,impulse)));b.v=sadd(b.v,smul(c.invMassB,impulse));b.w=sadd(b.w,smulm(c.invIB,scross(cp.rB,impulse)));}
    if(p.useBias!=0u)continue;float twistSpeed=sdot(m.normal,ssub(b.w,a.w));float maxTwist=c.friction*totalTwist;float oldTwist=m.twistImpulse;m.twistImpulse=clamp(oldTwist-m.twistMass*twistSpeed,-maxTwist,maxTwist);
    S3 twist=smul(m.twistImpulse-oldTwist,m.normal);a.w=ssub(a.w,smulm(c.invIA,twist));b.w=sadd(b.w,smulm(c.invIB,twist));
    if(c.rollingResistance>0){S3 rd=smulm(c.rollingMass,ssub(a.w,b.w));S3 old=m.rollingImpulse;m.rollingImpulse=sadd(old,rd);float limit=c.rollingResistance*totalNormal;float mag2=sdot(m.rollingImpulse,m.rollingImpulse);if(mag2>limit*limit+1.1920929e-7f)m.rollingImpulse=smul(limit/sqrt(mag2),m.rollingImpulse);rd=ssub(m.rollingImpulse,old);a.w=ssub(a.w,smulm(c.invIA,rd));b.w=sadd(b.w,smulm(c.invIB,rd));}
    S3 vrA=sadd(a.v,scross(a.w,m.centerA));S3 vrB=sadd(b.v,scross(b.w,m.centerB));S3 vr=ssub(vrB,vrA);float vx=sdot(vr,m.tangent1)-m.tangentVelocity1;float vy=sdot(vr,m.tangent2)-m.tangentVelocity2;
    float dx=-(m.tangentMass.cx.x*vx+m.tangentMass.cy.x*vy);float dy=-(m.tangentMass.cx.y*vx+m.tangentMass.cy.y*vy);float oldX=m.frictionImpulse.x,oldY=m.frictionImpulse.y;float nx=oldX+dx,ny=oldY+dy;float limit=c.friction*totalNormal;float len2=nx*nx+ny*ny;if(len2>limit*limit){float scale=limit/sqrt(len2);nx*=scale;ny*=scale;}dx=nx-oldX;dy=ny-oldY;m.frictionImpulse=S2{nx,ny};
    S3 impulse=sadd(smul(dx,m.tangent1),smul(dy,m.tangent2));a.v=ssub(a.v,smul(c.invMassA,impulse));a.w=ssub(a.w,smulm(c.invIA,scross(m.centerA,impulse)));b.v=sadd(b.v,smul(c.invMassB,impulse));b.w=sadd(b.w,smulm(c.invIB,scross(m.centerB,impulse)));
  }store_body(states,c.indexA,a);store_body(states,c.indexB,b);}
kernel void b3_solve_mesh(device BodyState* s [[buffer(0)]],device MeshContact* c [[buffer(1)]],device MeshManifold* m [[buffer(2)]],constant ContactParams& p [[buffer(3)]],uint tid [[thread_position_in_grid]]){if(tid<p.count)solve_mesh_one(s,c,m,p,p.offset+tid);}
kernel void b3_solve_mesh_overflow(device BodyState* s [[buffer(0)]],device MeshContact* c [[buffer(1)]],device MeshManifold* m [[buffer(2)]],constant ContactParams& p [[buffer(3)]]){for(uint i=0;i<p.count;++i)solve_mesh_one(s,c,m,p,p.offset+i);}
void restitution_mesh_one(device BodyState* states,device MeshContact* contacts,device MeshManifold* manifolds,constant ContactParams& p,uint index){
  device MeshContact& c=contacts[index];if(c.restitution==0)return;BodyS a=load_body(states,c.indexA);BodyS b=load_body(states,c.indexB);
  for(int mi=0;mi<c.manifoldCount;++mi){device MeshManifold& m=manifolds[c.manifoldStart+mi];for(int j=0;j<m.pointCount;++j){device MeshPoint& cp=m.points[j];if(cp.relativeVelocity>-p.restitutionThreshold||cp.totalNormalImpulse==0)continue;
    S3 vrA=sadd(a.v,scross(a.w,cp.rA));S3 vrB=sadd(b.v,scross(b.w,cp.rB));float vn=sdot(ssub(vrB,vrA),m.normal);float impulse=-cp.normalMass*(vn+c.restitution*cp.relativeVelocity);float next=max(cp.normalImpulse+impulse,0.0f);impulse=next-cp.normalImpulse;cp.normalImpulse=next;cp.totalNormalImpulse+=impulse;
    S3 P=smul(impulse,m.normal);a.v=ssub(a.v,smul(c.invMassA,P));a.w=ssub(a.w,smulm(c.invIA,scross(cp.rA,P)));b.v=sadd(b.v,smul(c.invMassB,P));b.w=sadd(b.w,smulm(c.invIB,scross(cp.rB,P)));}
  }store_body(states,c.indexA,a);store_body(states,c.indexB,b);
}
kernel void b3_restitution_mesh(device BodyState* s [[buffer(0)]],device MeshContact* c [[buffer(1)]],device MeshManifold* m [[buffer(2)]],constant ContactParams& p [[buffer(3)]],uint tid [[thread_position_in_grid]]){if(tid<p.count)restitution_mesh_one(s,c,m,p,p.offset+tid);}
kernel void b3_restitution_mesh_overflow(device BodyState* s [[buffer(0)]],device MeshContact* c [[buffer(1)]],device MeshManifold* m [[buffer(2)]],constant ContactParams& p [[buffer(3)]]){for(uint i=0;i<p.count;++i)restitution_mesh_one(s,c,m,p,p.offset+i);}
void distance_apply(thread BodyS& a,thread BodyS& b,device DistanceJoint& j,S3 rA,S3 rB,S3 axis,float impulse){
  S3 P=smul(impulse,axis);a.v=ssub(a.v,smul(j.invMassA,P));a.w=ssub(a.w,smulm(j.invIA,scross(rA,P)));
  b.v=sadd(b.v,smul(j.invMassB,P));b.w=sadd(b.w,smulm(j.invIB,scross(rB,P)));}
void warm_distance_one(device BodyState* states,device DistanceJoint* joints,uint index){device DistanceJoint& j=joints[index];
  BodyS a=load_body(states,j.indexA),b=load_body(states,j.indexB);S3 rA=srotate(a.dqv,a.dqs,j.anchorA),rB=srotate(b.dqv,b.dqs,j.anchorB);
  S3 separation=sadd(j.deltaCenter,sadd(ssub(b.dp,a.dp),ssub(rB,rA)));float len=sqrt(sdot(separation,separation));S3 axis=len>0?smul(1.0f/len,separation):S3{0,0,0};
  distance_apply(a,b,j,rA,rB,axis,j.impulse+j.lowerImpulse-j.upperImpulse+j.motorImpulse);store_body(states,j.indexA,a);store_body(states,j.indexB,b);}
void solve_distance_one(device BodyState* states,device DistanceJoint* joints,constant JointParams& p,uint index){device DistanceJoint& j=joints[index];
  BodyS a=load_body(states,j.indexA),b=load_body(states,j.indexB);S3 rA=srotate(a.dqv,a.dqs,j.anchorA),rB=srotate(b.dqv,b.dqs,j.anchorB);
  S3 separation=sadd(j.deltaCenter,sadd(ssub(b.dp,a.dp),ssub(rB,rA)));float len=sqrt(sdot(separation,separation));S3 axis=len>0?smul(1.0f/len,separation):S3{0,0,0};
  bool soft=(j.flags&1u)!=0u&&(j.minLength<j.maxLength||(j.flags&2u)==0u);
  if(soft){if(j.hertz>0){S3 vr=sadd(ssub(b.v,a.v),ssub(scross(b.w,rB),scross(a.w,rA)));float cdot=sdot(axis,vr),C=len-j.length;
      float old=j.impulse;float impulse=-j.distanceSoftness.massScale*j.axialMass*(cdot+j.distanceSoftness.biasRate*C)-j.distanceSoftness.impulseScale*old;
      j.impulse=clamp(old+impulse,j.lowerSpringForce*p.h,j.upperSpringForce*p.h);distance_apply(a,b,j,rA,rB,axis,j.impulse-old);}
    if((j.flags&2u)!=0u){S3 vr=sadd(ssub(b.v,a.v),ssub(scross(b.w,rB),scross(a.w,rA)));float cdot=sdot(axis,vr),C=len-j.minLength,bias=0,ms=1,is=0;
      if(C>0)bias=C*p.invH;else if(p.useBias!=0u){bias=j.constraintSoftness.biasRate*C;ms=j.constraintSoftness.massScale;is=j.constraintSoftness.impulseScale;}
      float impulse=-ms*j.axialMass*(cdot+bias)-is*j.lowerImpulse;float next=max(0.0f,j.lowerImpulse+impulse);impulse=next-j.lowerImpulse;j.lowerImpulse=next;distance_apply(a,b,j,rA,rB,axis,impulse);
      vr=sadd(ssub(a.v,b.v),ssub(scross(a.w,rA),scross(b.w,rB)));cdot=sdot(axis,vr);C=j.maxLength-len;bias=0;ms=1;is=0;
      if(C>0)bias=C*p.invH;else if(p.useBias!=0u){bias=j.constraintSoftness.biasRate*C;ms=j.constraintSoftness.massScale;is=j.constraintSoftness.impulseScale;}
      impulse=-ms*j.axialMass*(cdot+bias)-is*j.upperImpulse;next=max(0.0f,j.upperImpulse+impulse);impulse=next-j.upperImpulse;j.upperImpulse=next;distance_apply(a,b,j,rA,rB,axis,-impulse);}
    if((j.flags&4u)!=0u){S3 vr=sadd(ssub(b.v,a.v),ssub(scross(b.w,rB),scross(a.w,rA)));float impulse=j.axialMass*(j.motorSpeed-sdot(axis,vr));
      float old=j.motorImpulse,maxImpulse=p.h*j.maxMotorForce;j.motorImpulse=clamp(old+impulse,-maxImpulse,maxImpulse);distance_apply(a,b,j,rA,rB,axis,j.motorImpulse-old);}
  }else{S3 vr=sadd(ssub(b.v,a.v),ssub(scross(b.w,rB),scross(a.w,rA)));float cdot=sdot(axis,vr),C=len-j.length,bias=0,ms=1,is=0;
    if(p.useBias!=0u){bias=j.constraintSoftness.biasRate*C;ms=j.constraintSoftness.massScale;is=j.constraintSoftness.impulseScale;}
    float impulse=-ms*j.axialMass*(cdot+bias)-is*j.impulse;j.impulse+=impulse;distance_apply(a,b,j,rA,rB,axis,impulse);}
  store_body(states,j.indexA,a);store_body(states,j.indexB,b);}
kernel void b3_warm_start_distance(device BodyState* s [[buffer(0)]],device DistanceJoint* j [[buffer(1)]],constant JointParams& p [[buffer(2)]],uint tid [[thread_position_in_grid]]){if(tid<p.count)warm_distance_one(s,j,p.offset+tid);}
kernel void b3_solve_distance(device BodyState* s [[buffer(0)]],device DistanceJoint* j [[buffer(1)]],constant JointParams& p [[buffer(2)]],uint tid [[thread_position_in_grid]]){if(tid<p.count)solve_distance_one(s,j,p,p.offset+tid);}
SQ sqmul(SQ a,SQ b){return SQ{sadd(sadd(scross(a.v,b.v),smul(a.s,b.v)),smul(b.s,a.v)),a.s*b.s-sdot(a.v,b.v)};}
SQ sqinvmul(SQ a,SQ b){S3 nv=smul(-1.0f,a.v);return SQ{sadd(sadd(scross(nv,b.v),smul(a.s,b.v)),smul(b.s,nv)),a.s*b.s-sdot(nv,b.v)};}
void warm_parallel_one(device BodyState* states,device ParallelJoint* joints,uint index){device ParallelJoint& j=joints[index];BodyS a=load_body(states,j.indexA),b=load_body(states,j.indexB);
  S3 impulse=sadd(smul(j.perpImpulse.x,j.perpAxisX),smul(j.perpImpulse.y,j.perpAxisY));a.w=ssub(a.w,smulm(j.invIA,impulse));b.w=sadd(b.w,smulm(j.invIB,impulse));store_body(states,j.indexA,a);store_body(states,j.indexB,b);}
void solve_parallel_one(device BodyState* states,device ParallelJoint* joints,constant JointParams& p,uint index){device ParallelJoint& j=joints[index];BodyS a=load_body(states,j.indexA),b=load_body(states,j.indexB);
  SQ qa=sqmul(SQ{a.dqv,a.dqs},j.quatA),qb=sqmul(SQ{b.dqv,b.dqs},j.quatB);if(sdot(qa.v,qb.v)+qa.s*qb.s<0){qb.v=smul(-1.0f,qb.v);qb.s=-qb.s;}SQ rel=sqinvmul(qa,qb);
  if(j.fixedRotation==0u&&j.maxTorque>0){S3 ax=smul(0.5f,srotate(qa.v,qa.s,sadd(smul(rel.s,S3{1,0,0}),scross(rel.v,S3{1,0,0}))));S3 ay=smul(0.5f,srotate(qa.v,qa.s,sadd(smul(rel.s,S3{0,1,0}),scross(rel.v,S3{0,1,0}))));j.perpAxisX=ax;j.perpAxisY=ay;
    SM3 sum=SM3{sadd(j.invIA.cx,j.invIB.cx),sadd(j.invIA.cy,j.invIB.cy),sadd(j.invIA.cz,j.invIB.cz)};float kxx=sdot(ax,smulm(sum,ax)),kyy=sdot(ay,smulm(sum,ay)),kxy=sdot(ax,smulm(sum,ay));
    S3 wr=ssub(b.w,a.w);float bx=sdot(wr,ax)+j.softness.biasRate*rel.v.x,by=sdot(wr,ay)+j.softness.biasRate*rel.v.y;float det=kxx*kyy-kxy*kxy;float sx=0,sy=0;if(fabs(det)>1.17549435e-35f){sx=(kyy*bx-kxy*by)/det;sy=(kxx*by-kxy*bx)/det;}
    S2 old=j.perpImpulse;S2 next=S2{old.x-j.softness.massScale*sx-j.softness.impulseScale*old.x,old.y-j.softness.massScale*sy-j.softness.impulseScale*old.y};float maxImpulse=p.h*j.maxTorque,len2=next.x*next.x+next.y*next.y;if(len2>maxImpulse*maxImpulse){float scale=maxImpulse/sqrt(len2);next.x*=scale;next.y*=scale;}j.perpImpulse=next;
    S3 impulse=sadd(smul(next.x-old.x,ax),smul(next.y-old.y,ay));a.w=ssub(a.w,smulm(j.invIA,impulse));b.w=sadd(b.w,smulm(j.invIB,impulse));}
  store_body(states,j.indexA,a);store_body(states,j.indexB,b);}
kernel void b3_warm_start_parallel(device BodyState* s [[buffer(0)]],device ParallelJoint* j [[buffer(1)]],constant JointParams& p [[buffer(2)]],uint tid [[thread_position_in_grid]]){if(tid<p.count)warm_parallel_one(s,j,p.offset+tid);}
kernel void b3_solve_parallel(device BodyState* s [[buffer(0)]],device ParallelJoint* j [[buffer(1)]],constant JointParams& p [[buffer(2)]],uint tid [[thread_position_in_grid]]){if(tid<p.count)solve_parallel_one(s,j,p,p.offset+tid);}
kernel void b3_warm_start_joint_overflow(device BodyState* s [[buffer(0)]],device DistanceJoint* d [[buffer(1)]],device ParallelJoint* r [[buffer(2)]],device const JointOverflow* o [[buffer(3)]],constant JointParams& p [[buffer(4)]]){for(uint i=0;i<p.count;++i){JointOverflow e=o[i];if(e.type==0u)warm_distance_one(s,d,e.index);else warm_parallel_one(s,r,e.index);}}
kernel void b3_solve_joint_overflow(device BodyState* s [[buffer(0)]],device DistanceJoint* d [[buffer(1)]],device ParallelJoint* r [[buffer(2)]],device const JointOverflow* o [[buffer(3)]],constant JointParams& p [[buffer(4)]]){for(uint i=0;i<p.count;++i){JointOverflow e=o[i];if(e.type==0u)solve_distance_one(s,d,p,e.index);else solve_parallel_one(s,r,p,e.index);}}
