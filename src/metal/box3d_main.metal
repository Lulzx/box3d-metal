#include <metal_stdlib>
using namespace metal;
#include "metal_abi.h"
#if defined( B3_DOUBLE_PRECISION )
// Textual inclusion: the offline and runtime compilers see one translation
// unit, exactly like the historical concatenated source. The fallback
// generator splices the file contents here instead (runtime compile has no
// header search path); the false branch below needs no file lookup.
#include "metal_vf64_ieee.metal"
#endif
constant uint pair_custom_filter_bit=0x80000000u;
constant uint pair_type_mask=0x0fffffffu;
float4 quat_mul(float4 a, float4 b) {
  return float4(cross(a.xyz, b.xyz) + a.w * b.xyz + b.w * a.xyz,
                a.w * b.w - dot(a.xyz, b.xyz));
}
float3 inv_rotate(float4 q, float3 v) {
  float3 t1 = cross(q.xyz, v);
  float3 t2 = t1 - q.w * v;
  return v + 2.0f * cross(q.xyz, t2);
}
float3 rotate(float4 q, float3 v) {
  float3 t1 = cross(q.xyz, v);
  float3 t2 = t1 + q.w * v;
  return v + 2.0f * cross(q.xyz, t2);
}
#if defined(B3_DOUBLE_PRECISION)
ulong b3_vf64_add_float(ulong a, float b) {
  uint flags=0; ulong bb=soft_format_to_f64_status(ulong(as_type<uint>(b)),8u,23u,127,flags);
  return soft_add64_status(a,bb,soft_round_near_even,flags);
}
float b3_vf64_bound(ulong center,float delta,float origin,float local,uint roundingMode) {
  ulong value=b3_vf64_add_float(center,delta); value=b3_vf64_add_float(value,origin);
  value=b3_vf64_add_float(value,local); uint flags=0;
  return as_type<float>(uint(soft_f64_to_format_status(value,roundingMode,8u,23u,127,flags)));
}
float b3_vf64_difference(ulong b,ulong a) {
  uint flags=0; ulong value=soft_sub64_status(b,a,soft_round_near_even,flags);
  return as_type<float>(uint(soft_f64_to_format_status(value,soft_round_near_even,8u,23u,127,flags)));
}
#endif
float3x3 invert_matrix(float3x3 m) {
  float det = dot(m[0], cross(m[1], m[2]));
  if (fabs(det) <= 1.17549435e-35f) return float3x3(0.0f);
  float invDet = 1.0f / det;
  float3x3 cof = float3x3(invDet * cross(m[1], m[2]),
                           invDet * cross(m[2], m[0]),
                           invDet * cross(m[0], m[1]));
  return transpose(cof);
}
float3 solve3(float3x3 m, float3 a) {
  float det = dot(m[0], cross(m[1], m[2]));
  if (fabs(det) <= 1.17549435e-35f) return float3(0.0f);
  float invDet = 1.0f / det;
  return float3(invDet * dot(cross(m[1], m[2]), a),
                invDet * dot(cross(m[2], m[0]), a),
                invDet * dot(cross(m[0], m[1]), a));
}
kernel void b3_integrate_unconstrained(device BodyState* states [[buffer(0)]],
                                        const device BodyProperties* props [[buffer(1)]],
                                        constant FusedParams& p [[buffer(2)]],
                                        uint i [[thread_position_in_grid]]) {
  if (i >= p.bodyCount) return;
  BodyState s = states[i];
  BodyProperties bp = props[i];
  float linearDamping = 1.0f / (1.0f + p.h * bp.linearDamping);
  float angularDamping = 1.0f / (1.0f + p.h * bp.angularDamping);
  float gravityScale = bp.invMass > 0.0f ? bp.gravityScale : 0.0f;
  float3 force = float3(bp.forceX, bp.forceY, bp.forceZ);
  float3 gravity = float3(p.gravityX, p.gravityY, p.gravityZ);
  float3 forceStep = p.h * bp.invMass * force + p.h * gravityScale * gravity;
  float3x3 invIW = float3x3(
    float3(bp.invInertiaWorld[0], bp.invInertiaWorld[1], bp.invInertiaWorld[2]),
    float3(bp.invInertiaWorld[3], bp.invInertiaWorld[4], bp.invInertiaWorld[5]),
    float3(bp.invInertiaWorld[6], bp.invInertiaWorld[7], bp.invInertiaWorld[8]));
  float3 torque = float3(bp.torqueX, bp.torqueY, bp.torqueZ);
  float3 torqueStep = p.h * (invIW * torque);
  float4 q0 = float4(bp.qx, bp.qy, bp.qz, bp.qw);
  float4 dq = float4(s.qx, s.qy, s.qz, s.qw);
  float3 v = float3(s.lvx, s.lvy, s.lvz);
  float3 w = float3(s.avx, s.avy, s.avz);
  float3x3 invIL = float3x3(
    float3(bp.invInertiaLocal[0], bp.invInertiaLocal[1], bp.invInertiaLocal[2]),
    float3(bp.invInertiaLocal[3], bp.invInertiaLocal[4], bp.invInertiaLocal[5]),
    float3(bp.invInertiaLocal[6], bp.invInertiaLocal[7], bp.invInertiaLocal[8]));
  float3x3 inertia = invert_matrix(invIL);
  float i00 = inertia[0].x, i01 = inertia[1].x, i02 = inertia[2].x;
  float i11 = inertia[1].y, i12 = inertia[2].y, i22 = inertia[2].z;
  float maxV2 = p.maxLinearSpeed * p.maxLinearSpeed;
  float maxW2 = p.maxAngularSpeed * p.maxAngularSpeed;
  uint steps = max(p.subStepCount, 1u);
  for (uint step = 0u; step < steps; ++step) {
  v = forceStep + linearDamping * v;
  w = torqueStep + angularDamping * w;
  float4 q = quat_mul(dq, q0);
  float3 omega1 = inv_rotate(q, w);
  float3 omega2 = omega1;
  float w1 = omega2.x, w2 = omega2.y, w3 = omega2.z;
  float Iw1 = i00*w1 + i01*w2 + i02*w3;
  float Iw2 = i01*w1 + i11*w2 + i12*w3;
  float Iw3 = i02*w1 + i12*w2 + i22*w3;
  float3 residual = float3(
    p.h * (w2*Iw3 - w3*Iw2),
    p.h * (w3*Iw1 - w1*Iw3),
    p.h * (w1*Iw2 - w2*Iw1));
  float3x3 J = float3x3(
    float3(i00 + p.h*(w2*i02 - w3*i01),
           i01 + p.h*(w3*i00 - w1*i02 - Iw3),
           i02 + p.h*(w1*i01 - w2*i00 + Iw2)),
    float3(i01 + p.h*(w2*i12 - w3*i11 + Iw3),
           i11 + p.h*(w3*i01 - w1*i12),
           i12 + p.h*(w1*i11 - w2*i01 - Iw1)),
    float3(i02 + p.h*(w2*i22 - w3*i12 - Iw2),
           i12 + p.h*(w3*i02 - w1*i22 + Iw1),
           i22 + p.h*(w1*i12 - w2*i02)));
  omega2 -= solve3(J, residual);
  w = rotate(q, omega2);
  if (s.flags & 0x00000001u) v.x = 0.0f;
  if (s.flags & 0x00000002u) v.y = 0.0f;
  if (s.flags & 0x00000004u) v.z = 0.0f;
  if (s.flags & 0x00000008u) w.x = 0.0f;
  if (s.flags & 0x00000010u) w.y = 0.0f;
  if (s.flags & 0x00000020u) w.z = 0.0f;
  float v2 = dot(v, v);
  if (v2 > maxV2) { v *= p.maxLinearSpeed / sqrt(v2); s.flags |= 0x00000100u; }
  float w2n = dot(w, w);
  if (w2n > maxW2 && (s.flags & 0x00000400u) == 0u) {
    w *= p.maxAngularSpeed / sqrt(w2n); s.flags |= 0x00000100u;
  }
  if (p.integratePosition != 0u) {
    float3 dp = float3(s.dpx, s.dpy, s.dpz) + p.h * v;
    float3 qdv = 0.5f * p.h * w;
    float3 qv = dq.xyz + cross(qdv, dq.xyz) + dq.w * qdv;
    float qw = dq.w - dot(qdv, dq.xyz);
    float q2 = dot(qv, qv) + qw * qw;
    if (q2 > 1.17549435e-35f) { float invQ = 1.0f / sqrt(q2); qv *= invQ; qw *= invQ; }
    else { qv = float3(0.0f); qw = 1.0f; }
    s.dpx = dp.x; s.dpy = dp.y; s.dpz = dp.z;
    dq = float4(qv.x, qv.y, qv.z, qw);
  }
  }
  s.lvx = v.x; s.lvy = v.y; s.lvz = v.z;
  s.avx = w.x; s.avy = w.y; s.avz = w.z;
  s.qx = dq.x; s.qy = dq.y; s.qz = dq.z; s.qw = dq.w;
  states[i] = s;
}
kernel void b3_integrate_positions(device BodyState* states [[buffer(0)]],
                                   constant Params& p [[buffer(1)]],
                                   uint i [[thread_position_in_grid]]) {
  if (i >= p.bodyCount) return;
  BodyState s = states[i];
  float3 v = float3(s.lvx, s.lvy, s.lvz);
  float3 w = float3(s.avx, s.avy, s.avz);
  if (s.flags & 0x00000001u) v.x = 0.0f;
  if (s.flags & 0x00000002u) v.y = 0.0f;
  if (s.flags & 0x00000004u) v.z = 0.0f;
  if (s.flags & 0x00000008u) w.x = 0.0f;
  if (s.flags & 0x00000010u) w.y = 0.0f;
  if (s.flags & 0x00000020u) w.z = 0.0f;
  float v2 = dot(v, v);
  float maxV2 = p.maxLinearSpeed * p.maxLinearSpeed;
  if (v2 > maxV2) { v *= p.maxLinearSpeed / sqrt(v2); s.flags |= 0x00000100u; }
  float w2 = dot(w, w);
  float maxW2 = p.maxAngularSpeed * p.maxAngularSpeed;
  if (w2 > maxW2 && (s.flags & 0x00000400u) == 0u) {
    w *= p.maxAngularSpeed / sqrt(w2); s.flags |= 0x00000100u;
  }
  float3 dp = float3(s.dpx, s.dpy, s.dpz) + p.h * v;
  float4 q = float4(s.qx, s.qy, s.qz, s.qw);
  float3 qdv = 0.5f * p.h * w;
  float3 qv = q.xyz + cross(qdv, q.xyz) + q.w * qdv;
  float qw = q.w - dot(qdv, q.xyz);
  float q2 = dot(qv, qv) + qw * qw;
  if (q2 > 1.17549435e-35f) { float invQ = 1.0f / sqrt(q2); qv *= invQ; qw *= invQ; }
  else { qv = float3(0.0f); qw = 1.0f; }
  s.lvx = v.x; s.lvy = v.y; s.lvz = v.z;
  s.avx = w.x; s.avy = w.y; s.avz = w.z;
  s.dpx = dp.x; s.dpy = dp.y; s.dpz = dp.z;
  s.qx = qv.x; s.qy = qv.y; s.qz = qv.z; s.qw = qw;
  states[i] = s;
}
kernel void b3_finalize_bodies(device BodyState* states [[buffer(0)]],
                               device BodyProperties* props [[buffer(1)]],
                               device FinalizeProperties* finalizeProps [[buffer(2)]],
                               device FinalizeResult* results [[buffer(3)]],
                               constant FinalizeParams& p [[buffer(4)]],
                               device BodyTransform* bodyTransforms [[buffer(5)]],
                               device BodyMoveResult* moveResults [[buffer(6)]],
                               uint i [[thread_position_in_grid]]) {
  if (i >= p.bodyCount) return;
  BodyState s = states[i]; BodyProperties bp = props[i]; FinalizeProperties fp = finalizeProps[i];
  float4 baseQ = float4(bp.qx, bp.qy, bp.qz, bp.qw);
  float4 dq = float4(s.qx, s.qy, s.qz, s.qw);
  float4 q = quat_mul(dq, baseQ);
  float q2 = dot(q, q);
  q = q2 > 1.17549435e-35f ? q * rsqrt(q2) : float4(0.0f, 0.0f, 0.0f, 1.0f);
  float3 v = float3(s.lvx, s.lvy, s.lvz);
  float3 w = float3(s.avx, s.avy, s.avz);
  float3 localOmega = abs(inv_rotate(baseQ, w));
  float3 localDelta = abs(inv_rotate(baseQ, dq.xyz));
  float3 extent = float3(fp.maxExtentX, fp.maxExtentY, fp.maxExtentZ);
  float3 velocityArc = float3(localOmega.y*extent.z + localOmega.z*extent.y,
                              localOmega.z*extent.x + localOmega.x*extent.z,
                              localOmega.x*extent.y + localOmega.y*extent.x);
  float3 rotationArc = float3(localDelta.y*extent.z + localDelta.z*extent.y,
                              localDelta.z*extent.x + localDelta.x*extent.z,
                              localDelta.x*extent.y + localDelta.y*extent.x);
  float maxVelocity = length(v) + length(velocityArc);
  float3 dp = float3(s.dpx, s.dpy, s.dpz);
  float maxDelta = length(dp) + 2.0f * length(rotationArc);
  float sleepVelocity = max(maxVelocity, 0.5f * p.invTimeStep * maxDelta);
  float3 localCenter = float3(fp.localCenterX, fp.localCenterY, fp.localCenterZ);
  float3 origin = -rotate(q, localCenter);
  float3 position = float3(fp.centerX,fp.centerY,fp.centerZ) + dp + origin;
  float3x3 R = float3x3(rotate(q, float3(1.0f,0.0f,0.0f)),
                          rotate(q, float3(0.0f,1.0f,0.0f)),
                          rotate(q, float3(0.0f,0.0f,1.0f)));
  float3x3 invIL = float3x3(
    float3(bp.invInertiaLocal[0],bp.invInertiaLocal[1],bp.invInertiaLocal[2]),
    float3(bp.invInertiaLocal[3],bp.invInertiaLocal[4],bp.invInertiaLocal[5]),
    float3(bp.invInertiaLocal[6],bp.invInertiaLocal[7],bp.invInertiaLocal[8]));
  float3x3 invIW = R * invIL * transpose(R);
  FinalizeResult out; out.dpx=dp.x; out.dpy=dp.y; out.dpz=dp.z;
  out.qx=q.x; out.qy=q.y; out.qz=q.z; out.qw=q.w;
  out.originX=origin.x; out.originY=origin.y; out.originZ=origin.z;
  out.positionX=position.x; out.positionY=position.y; out.positionZ=position.z;
  out.sleepVelocity=sleepVelocity; out.maxVelocity=maxVelocity; out.maxDeltaPosition=maxDelta;
  out.invInertiaWorld[0]=invIW[0].x; out.invInertiaWorld[1]=invIW[0].y; out.invInertiaWorld[2]=invIW[0].z;
  out.invInertiaWorld[3]=invIW[1].x; out.invInertiaWorld[4]=invIW[1].y; out.invInertiaWorld[5]=invIW[1].z;
  out.invInertiaWorld[6]=invIW[2].x; out.invInertiaWorld[7]=invIW[2].y; out.invInertiaWorld[8]=invIW[2].z;
  results[i] = out;
  if(p.publishTransforms!=0u&&fp.bodyId>=0&&uint(fp.bodyId)<p.transformCount){
    BodyTransform t=bodyTransforms[fp.bodyId];t.qx=q.x;t.qy=q.y;t.qz=q.z;t.qw=q.w;
    t.px=position.x;t.py=position.y;t.pz=position.z;t.supported=1u;t.index=int(i);t.flags=s.flags;
    t.localCenterX=fp.localCenterX;t.localCenterY=fp.localCenterY;t.localCenterZ=fp.localCenterZ;
    t.sleepVelocity=sleepVelocity;
    BodyMoveResult move;move.qx=q.x;move.qy=q.y;move.qz=q.z;move.qw=q.w;
    move.px=position.x;move.py=position.y;move.pz=position.z;move.bodyId=fp.bodyId;
    move.userData=fp.userData;move.generationWorld=fp.generationWorld;move.padding=0u;
    bp.qx=q.x;bp.qy=q.y;bp.qz=q.z;bp.qw=q.w;
    bp.forceX=0.0f;bp.forceY=0.0f;bp.forceZ=0.0f;bp.torqueX=0.0f;bp.torqueY=0.0f;bp.torqueZ=0.0f;
    bp.invInertiaWorld[0]=invIW[0].x;bp.invInertiaWorld[1]=invIW[0].y;bp.invInertiaWorld[2]=invIW[0].z;
    bp.invInertiaWorld[3]=invIW[1].x;bp.invInertiaWorld[4]=invIW[1].y;bp.invInertiaWorld[5]=invIW[1].z;
    bp.invInertiaWorld[6]=invIW[2].x;bp.invInertiaWorld[7]=invIW[2].y;bp.invInertiaWorld[8]=invIW[2].z;props[i]=bp;
    fp.centerX+=dp.x;fp.centerY+=dp.y;fp.centerZ+=dp.z;
#if defined(B3_DOUBLE_PRECISION)
    fp.centerXBits=b3_vf64_add_float(fp.centerXBits,dp.x);
    fp.centerYBits=b3_vf64_add_float(fp.centerYBits,dp.y);
    fp.centerZBits=b3_vf64_add_float(fp.centerZBits,dp.z);
    t.pxBits=b3_vf64_add_float(fp.centerXBits,origin.x);
    t.pyBits=b3_vf64_add_float(fp.centerYBits,origin.y);
    t.pzBits=b3_vf64_add_float(fp.centerZBits,origin.z);
    move.pxBits=t.pxBits;move.pyBits=t.pyBits;move.pzBits=t.pzBits;
#else
    move.pxBits=0ul;move.pyBits=0ul;move.pzBits=0ul;
#endif
    finalizeProps[i]=fp;bodyTransforms[fp.bodyId]=t;moveResults[i]=move;
    s.dpx=0.0f;s.dpy=0.0f;s.dpz=0.0f;s.qx=0.0f;s.qy=0.0f;s.qz=0.0f;s.qw=1.0f;
    s.flags&=~0x00000340u;states[i]=s;
  }
}
kernel void b3_finalize_shapes(const device ShapeInput* inputs [[buffer(0)]],
                               const device FinalizeResult* bodies [[buffer(1)]],
                               device ShapeResult* results [[buffer(2)]],
                               constant ShapeParams& p [[buffer(3)]],
                               const device FinalizeProperties* finalizeProps [[buffer(4)]],
                               uint i [[thread_position_in_grid]]) {
  if (i >= p.shapeCount) return; ShapeInput in = inputs[i]; FinalizeResult b = bodies[in.bodyIndex];
  FinalizeProperties fp=finalizeProps[in.bodyIndex];
  float4 q=float4(b.qx,b.qy,b.qz,b.qw); float3 localLo,localHi;
  if (in.type == 5u) { float3 c=rotate(q,float3(in.p1x,in.p1y,in.p1z));
    localLo=c-float3(in.radius); localHi=c+float3(in.radius);
  } else if (in.type == 0u) {
    float3 a=rotate(q,float3(in.p1x,in.p1y,in.p1z));
    float3 c=rotate(q,float3(in.p2x,in.p2y,in.p2z));
    localLo=min(a,c)-float3(in.radius); localHi=max(a,c)+float3(in.radius);
  } else {
    float3 sourceLo=float3(in.p1x,in.p1y,in.p1z), sourceHi=float3(in.p2x,in.p2y,in.p2z);
    float3 center=0.5f*(sourceLo+sourceHi), extent=0.5f*(sourceHi-sourceLo);
    float3 rx=rotate(q,float3(1,0,0)), ry=rotate(q,float3(0,1,0)), rz=rotate(q,float3(0,0,1));
    float3 rotatedCenter=rotate(q,center); float3 worldExtent=abs(rx)*extent.x+abs(ry)*extent.y+abs(rz)*extent.z;
    localLo=rotatedCenter-worldExtent; localHi=rotatedCenter+worldExtent;
  }
#if defined(B3_DOUBLE_PRECISION)
  float3 arithmeticScale=max(float3(1.0f),max(abs(localLo),abs(localHi)));
  float3 arithmeticSlack=1.9073486328125e-6f*arithmeticScale;
  localLo-=arithmeticSlack; localHi+=arithmeticSlack;
#endif
  localLo-=float3(p.extra); localHi+=float3(p.extra);
#if defined(B3_DOUBLE_PRECISION)
  float3 lo=float3(b3_vf64_bound(fp.centerXBits,0.0f,b.originX,localLo.x,soft_round_min),
                   b3_vf64_bound(fp.centerYBits,0.0f,b.originY,localLo.y,soft_round_min),
                   b3_vf64_bound(fp.centerZBits,0.0f,b.originZ,localLo.z,soft_round_min));
  float3 hi=float3(b3_vf64_bound(fp.centerXBits,0.0f,b.originX,localHi.x,soft_round_max),
                   b3_vf64_bound(fp.centerYBits,0.0f,b.originY,localHi.y,soft_round_max),
                   b3_vf64_bound(fp.centerZBits,0.0f,b.originZ,localHi.z,soft_round_max));
#else
  float3 position=float3(b.positionX,b.positionY,b.positionZ);
  float3 lo=position+localLo, hi=position+localHi;
#endif
  float3 oldLo,oldHi;if(p.useResidentBounds!=0u){ShapeResult previous=results[i];
    oldLo=float3(previous.flx,previous.fly,previous.flz);oldHi=float3(previous.fux,previous.fuy,previous.fuz);
  }else{oldLo=float3(in.oldLx,in.oldLy,in.oldLz);oldHi=float3(in.oldUx,in.oldUy,in.oldUz);}
  bool enlarged=any(oldLo>lo)||any(hi>oldHi); float3 fatLo=oldLo, fatHi=oldHi;
  if (enlarged) { fatLo=lo-float3(in.margin); fatHi=hi+float3(in.margin); }
  ShapeResult out; out.shapeId=in.shapeId; out.bodyIndex=in.bodyIndex; out.enlarged=enlarged?1u:0u; out.padding=0;
  out.lx=lo.x;out.ly=lo.y;out.lz=lo.z;out.ux=hi.x;out.uy=hi.y;out.uz=hi.z;
  out.flx=fatLo.x;out.fly=fatLo.y;out.flz=fatLo.z;out.fux=fatHi.x;out.fuy=fatHi.y;out.fuz=fatHi.z; results[i]=out;
}
kernel void b3_shape_scan_blocks(device ShapeResult* results [[buffer(0)]],device PairBlock* blocks [[buffer(1)]],
                                 constant ShapeCompactParams& p [[buffer(2)]],uint i [[thread_position_in_grid]],
                                 uint threadIndex [[thread_index_in_threadgroup]],uint group [[threadgroup_position_in_grid]],
                                 ushort lane [[thread_index_in_simdgroup]],ushort subgroup [[simdgroup_index_in_threadgroup]],
                                 ushort simdWidth [[threads_per_simdgroup]]) {
  threadgroup uint subgroupTotals[32];threadgroup uint subgroupOffsets[32];
  uint value=i<p.shapeCount?results[i].enlarged:0u;uint localOffset=simd_prefix_exclusive_sum(value);
  uint subgroupTotal=simd_sum(value);if(lane==0){subgroupTotals[subgroup]=subgroupTotal;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(threadIndex==0u){uint running=0u;uint subgroupCount=256u/uint(simdWidth);
    for(uint s=0u;s<subgroupCount;++s){subgroupOffsets[s]=running;running+=subgroupTotals[s];}
    PairBlock b;b.sum=running;b.flags=0u;b.offset=0u;b.padding=0u;blocks[group]=b;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(i<p.shapeCount){ShapeResult r=results[i];r.padding=localOffset+subgroupOffsets[subgroup];results[i]=r;}
}
kernel void b3_shape_prefix(device PairBlock* blocks [[buffer(0)]],device PairSummary* summary [[buffer(1)]],
                            constant ShapeCompactParams& p [[buffer(2)]]) {
  uint total=0u;for(uint i=0u;i<p.blockCount;++i){PairBlock b=blocks[i];b.offset=total;blocks[i]=b;total+=b.sum;}
  summary->totalCount=ulong(total);summary->flags=0u;summary->writeFlags=0u;summary->cpuFilterMoveCount=0u;summary->padding=0u;
}
kernel void b3_shape_scatter(const device ShapeInput* inputs [[buffer(0)]],const device ShapeResult* results [[buffer(1)]],
                             const device PairBlock* blocks [[buffer(2)]],device EnlargedShapeResult* compact [[buffer(3)]],
                             constant ShapeCompactParams& p [[buffer(4)]],uint i [[thread_position_in_grid]]) {
  if(i>=p.shapeCount)return;ShapeResult r=results[i];if(r.enlarged==0u)return;
  uint output=blocks[i/256u].offset+r.padding;ShapeInput input=inputs[i];EnlargedShapeResult c;
  c.shapeId=int(r.shapeId);c.proxyKey=input.proxyKey;c.lx=r.flx;c.ly=r.fly;c.lz=r.flz;
  c.ux=r.fux;c.uy=r.fuy;c.uz=r.fuz;compact[output]=c;
}
kernel void b3_pair_update_leaves(const device ShapeInput* inputs [[buffer(0)]],
                                  const device ShapeResult* results [[buffer(1)]],
                                  device TreeNode* nodes [[buffer(2)]],
                                  constant TreeOffsets& offsets [[buffer(3)]],uint i [[thread_position_in_grid]]) {
  ShapeInput in=inputs[i];ShapeResult r=results[i];if(r.enlarged==0u||in.proxyKey<0)return;
  uint type=uint(in.proxyKey)&3u;uint proxy=uint(in.proxyKey)>>2u;if(type>=3u)return;
  uint offset=type==0u?offsets.offset0:(type==1u?offsets.offset1:offsets.offset2);
  TreeNode n=nodes[offset+proxy];if((n.flags&4u)==0u)return;
  n.lx=r.flx;n.ly=r.fly;n.lz=r.flz;n.ux=r.fux;n.uy=r.fuy;n.uz=r.fuz;nodes[offset+proxy]=n;
}
kernel void b3_pair_refit(device TreeNode* nodes [[buffer(0)]],constant TreeRefitParams& p [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
  if(i>=p.nodeCount)return;uint index=p.nodeOffset+i;TreeNode n=nodes[index];
  if((n.flags&1u)==0u||(n.flags&4u)!=0u||uint(n.height)!=p.targetHeight)return;
  TreeNode a=nodes[p.nodeOffset+n.child1],b=nodes[p.nodeOffset+n.child2];
  n.lx=min(a.lx,b.lx);n.ly=min(a.ly,b.ly);n.lz=min(a.lz,b.lz);
  n.ux=max(a.ux,b.ux);n.uy=max(a.uy,b.uy);n.uz=max(a.uz,b.uz);nodes[index]=n;
}
inline bool tree_overlap(TreeNode n,float3 lo,float3 hi) {
  return !(n.ux<lo.x||n.lx>hi.x||n.uy<lo.y||n.ly>hi.y||n.uz<lo.z||n.lz>hi.z);
}
inline bool pair_shapes_collide(PairShape a,PairShape b) {
  if(a.bodyId==b.bodyId||a.sensorIndex>=0||b.sensorIndex>=0)return false;
  if(a.groupIndex==b.groupIndex&&a.groupIndex!=0)return a.groupIndex>0;
  return (a.maskBits&b.categoryBits)!=0ul&&(a.categoryBits&b.maskBits)!=0ul;
}
inline uint pair_key_hash(ulong key) {
  ulong h=key;h^=h>>33;h*=0xff51afd7ed558ccdul;h^=h>>33;
  h*=0xc4ceb9fe1a85ec53ul;h^=h>>33;return uint(h);
}
inline bool pair_set_contains_key(const device SetItem* items,uint capacity,ulong key) {
  uint index=pair_key_hash(key)&(capacity-1u);
  for(uint probe=0u;probe<capacity;++probe){SetItem item=items[index];
    if(item.hash==0u)return false;if(item.key==key)return true;index=(index+1u)&(capacity-1u);}
  return false;
}
inline bool pair_set_contains(const device SetItem* items,uint capacity,uint shapeA,uint shapeB) {
  uint lo=min(shapeA,shapeB),hi=max(shapeA,shapeB);
  ulong key=(ulong(lo&0x3fffffu)<<42)|(ulong(hi&0x3fffffu)<<20);
  return pair_set_contains_key(items,capacity,key);
}
inline bool body_pair_set_contains(const device SetItem* items,uint capacity,int bodyA,int bodyB) {
  uint a=uint(bodyA)+1u,b=uint(bodyB)+1u,lo=min(a,b),hi=max(a,b);
  return pair_set_contains_key(items,capacity,(ulong(lo)<<32)|ulong(hi));
}
inline void query_pair_tree(const device TreeNode* nodes,const device uint* moved,const device PairShape* shapes,
                            const device SetItem* pairSet,const device SetItem* filterSet,uint filterCapacity,
                            uint shapeCount,uint pairCapacity,uint queryShapeIndex,
                            PairShape queryShape,int root,uint nodeOffset,int treeType,
                            int queryKey,int queryType,uint movedEpoch,float3 lo,float3 hi,
                            device PairCandidate* candidates,uint outputOffset,uint expected,uint writeCandidates,
                            thread int* stack,thread uint& candidateCount,thread uint& queryFlags,
                            thread uint& cpuFilter) {
  if(root<0||queryFlags!=0u) return; uint stackCount=0u; stack[stackCount++]=root;
  while(stackCount>0u) { int nodeId=stack[--stackCount]; TreeNode n=nodes[nodeOffset+uint(nodeId)];
    if(n.categoryBits==0ul||!tree_overlap(n,lo,hi)) continue;
    if((n.flags&4u)!=0u) {
      int targetKey=(nodeId<<2)|treeType;if(targetKey==queryKey)continue;
      bool targetMoved=moved[nodeOffset+uint(nodeId)]==movedEpoch;
      if((queryType==2&&treeType==2&&targetKey<queryKey&&targetMoved)||(queryType!=2&&targetMoved))continue;
      uint shapeIndex=n.child1;if(shapeIndex>=shapeCount||shapes[shapeIndex].bodyId<0){queryFlags|=16u;return;}
      if(!pair_shapes_collide(queryShape,shapes[shapeIndex]))continue;
      if(body_pair_set_contains(filterSet,filterCapacity,queryShape.bodyId,shapes[shapeIndex].bodyId))continue;
      PairShape targetShape=shapes[shapeIndex];uint targetType=targetShape.type&pair_type_mask;
      if(targetType!=1u&&pair_set_contains(pairSet,pairCapacity,queryShapeIndex,shapeIndex))continue;
      if(targetType==1u||((queryShape.type|targetShape.type)&pair_custom_filter_bit)!=0u)cpuFilter=1u;
      if(writeCandidates!=0u) { if(candidateCount>=expected) { queryFlags|=2u; return; }
        PairCandidate c; c.proxyId=nodeId;c.treeType=treeType;c.shapeIndex=int(n.child1);c.padding=0;
        candidates[outputOffset+candidateCount]=c; }
      candidateCount+=1u;
    } else { if(stackCount>=63u) { queryFlags|=1u; return; }
      stack[stackCount++]=int(n.child1); stack[stackCount++]=int(n.child2); }
  }
}
kernel void b3_pair_mark_moves(const device int* moves [[buffer(0)]],device uint* moved [[buffer(1)]],
                               const device EnlargedShapeResult* resident [[buffer(2)]],
                               constant PairParams& p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
  if(i>=p.moveCount)return;int key=p.residentMoves!=0u?resident[i].proxyKey:moves[i];
  uint type=uint(key)&3u;uint proxy=uint(key)>>2u;if(type>=3u)return;
  uint offset=type==0u?p.offset0:(type==1u?p.offset1:p.offset2);moved[offset+proxy]=p.movedEpoch;
}
kernel void b3_pair_candidates(const device int* moves [[buffer(0)]],const device TreeNode* nodes [[buffer(1)]],
                               device PairQueryRecord* records [[buffer(2)]],device PairCandidate* candidates [[buffer(3)]],
                               constant PairParams& p [[buffer(4)]],device PairSummary* summary [[buffer(5)]],
                               const device uint* moved [[buffer(6)]],
                               const device PairShape* shapes [[buffer(7)]],
                               const device SetItem* pairSet [[buffer(8)]],
                               const device EnlargedShapeResult* resident [[buffer(9)]],
                               const device SetItem* filterSet [[buffer(10)]],
                               uint i [[thread_position_in_grid]]) {
  if(i>=p.moveCount) return; int key=p.residentMoves!=0u?resident[i].proxyKey:moves[i];int queryType=key&3;int proxyId=key>>2;
  if(p.writeCandidates!=0u&&summary->flags!=0u) return;
  uint queryOffset=queryType==0?p.offset0:(queryType==1?p.offset1:p.offset2);
  TreeNode q=nodes[queryOffset+uint(proxyId)];float3 lo=float3(q.lx,q.ly,q.lz),hi=float3(q.ux,q.uy,q.uz);
  PairQueryRecord record;if(p.writeCandidates!=0u){record=records[i];}else{record.count=0u;record.offset=0u;record.cpuFilter=0u;record.cpuFilterOffset=0u;}
  uint count=0u,flags=0u,cpuFilter=0u;thread int stack[64];
  record.queryShapeIndex=int(q.child1);record.queryProxyKey=key;record.lx=q.lx;record.ly=q.ly;record.lz=q.lz;
  record.ux=q.ux;record.uy=q.uy;record.uz=q.uz;
  if(q.child1>=p.shapeCount||shapes[q.child1].bodyId<0){record.flags=16u;records[i]=record;return;}
  PairShape queryShape=shapes[q.child1];
  if((queryShape.type&pair_type_mask)==1u){record.flags=32u;records[i]=record;return;}
  if(queryType==2) {
    query_pair_tree(nodes,moved,shapes,pairSet,filterSet,p.filterCapacity,p.shapeCount,p.pairCapacity,q.child1,queryShape,p.root1,p.offset1,1,key,queryType,p.movedEpoch,lo,hi,candidates,record.offset,record.count,p.writeCandidates,stack,count,flags,cpuFilter);
    query_pair_tree(nodes,moved,shapes,pairSet,filterSet,p.filterCapacity,p.shapeCount,p.pairCapacity,q.child1,queryShape,p.root0,p.offset0,0,key,queryType,p.movedEpoch,lo,hi,candidates,record.offset,record.count,p.writeCandidates,stack,count,flags,cpuFilter);
  }
  query_pair_tree(nodes,moved,shapes,pairSet,filterSet,p.filterCapacity,p.shapeCount,p.pairCapacity,q.child1,queryShape,p.root2,p.offset2,2,key,queryType,p.movedEpoch,lo,hi,candidates,record.offset,record.count,p.writeCandidates,stack,count,flags,cpuFilter);
  if(p.writeCandidates==0u){record.count=count;record.cpuFilter=cpuFilter;}else if(count!=record.count||cpuFilter!=record.cpuFilter)flags|=2u;
  record.flags=flags;records[i]=record;
  if(p.writeCandidates==0u&&flags==0u){device atomic_uint* total=(device atomic_uint*)(cpuFilter!=0u?&summary->cpuFilterCandidateCount:&summary->directCandidateCount);
    atomic_fetch_add_explicit(total,count,memory_order_relaxed);}
  if(p.writeCandidates!=0u&&flags!=0u) atomic_fetch_or_explicit((device atomic_uint*)&summary->writeFlags,8u,memory_order_relaxed);
}
kernel void b3_pair_scan_blocks(device PairQueryRecord* records [[buffer(0)]],device PairBlock* blocks [[buffer(1)]],
                                constant PairPrefixParams& p [[buffer(2)]],uint i [[thread_position_in_grid]],
                                uint threadIndex [[thread_index_in_threadgroup]],uint group [[threadgroup_position_in_grid]],
                                ushort lane [[thread_index_in_simdgroup]],ushort subgroup [[simdgroup_index_in_threadgroup]],
                                ushort simdWidth [[threads_per_simdgroup]]) {
  threadgroup uint subgroupTotals[32],subgroupOffsets[32],subgroupErrors[32];
  threadgroup uint subgroupCpuTotals[32],subgroupCpuOffsets[32];
  uint value=0u,error=0u,cpu=0u;if(i<p.moveCount){PairQueryRecord r=records[i];value=r.count;error=r.flags;cpu=r.cpuFilter;}
  uint localOffset=simd_prefix_exclusive_sum(value),localCpuOffset=simd_prefix_exclusive_sum(cpu);
  uint subgroupTotal=simd_sum(value),subgroupCpuTotal=simd_sum(cpu),subgroupError=simd_max(error);
  if(lane==0){subgroupTotals[subgroup]=subgroupTotal;subgroupCpuTotals[subgroup]=subgroupCpuTotal;subgroupErrors[subgroup]=subgroupError;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(threadIndex==0u){uint running=0u,cpuRunning=0u,flags=0u;uint subgroupCount=256u/uint(simdWidth);
    for(uint s=0u;s<subgroupCount;++s){subgroupOffsets[s]=running;running+=subgroupTotals[s];
      subgroupCpuOffsets[s]=cpuRunning;cpuRunning+=subgroupCpuTotals[s];flags|=subgroupErrors[s];}
    PairBlock b;b.sum=running;b.flags=flags;b.offset=0u;b.padding=cpuRunning;blocks[group]=b;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(i<p.moveCount){PairQueryRecord r=records[i];r.offset=localOffset+subgroupOffsets[subgroup];
    r.cpuFilterOffset=localCpuOffset+subgroupCpuOffsets[subgroup];records[i]=r;}
}
kernel void b3_pair_prefix(device PairBlock* blocks [[buffer(0)]],device PairSummary* summary [[buffer(1)]],
                           constant PairPrefixParams& p [[buffer(2)]]) {
  ulong total=0ul;uint flags=0u,cpuTotal=0u;uint blockCount=(p.moveCount+255u)/256u;
  for(uint i=0u;i<blockCount;++i){PairBlock b=blocks[i];if(b.flags!=0u)flags|=1u;
    uint cpuCount=b.padding;b.offset=uint(min(total,ulong(0xffffffffu)));b.padding=cpuTotal;blocks[i]=b;
    total+=ulong(b.sum);cpuTotal+=cpuCount;}
  if(total>ulong(p.candidateLimit))flags|=2u;if(total>ulong(p.candidateCapacity))flags|=4u;
  summary->totalCount=total;summary->flags=flags;summary->writeFlags=0u;summary->cpuFilterMoveCount=cpuTotal;summary->padding=0u;
}
kernel void b3_pair_add_offsets(device PairQueryRecord* records [[buffer(0)]],const device PairBlock* blocks [[buffer(1)]],
                                constant PairPrefixParams& p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
  if(i<p.moveCount){PairQueryRecord r=records[i];PairBlock b=blocks[i/256u];r.offset+=b.offset;r.cpuFilterOffset+=b.padding;records[i]=r;}
}
kernel void b3_pair_compact_cpu_filter(const device PairQueryRecord* records [[buffer(0)]],device int* moves [[buffer(1)]],
                                       constant PairPrefixParams& p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
  if(i<p.moveCount&&records[i].cpuFilter!=0u)moves[records[i].cpuFilterOffset]=int(i);
}
kernel void b3_pair_contact_seeds(const device PairQueryRecord* records [[buffer(0)]],
                                  const device PairCandidate* candidates [[buffer(1)]],
                                  device PairContactSeed* seeds [[buffer(2)]],
                                  const device PairSummary* summary [[buffer(3)]],
                                  constant PairPrefixParams& p [[buffer(4)]],
                                  uint i [[thread_position_in_grid]]) {
  if(i>=p.moveCount||summary->flags!=0u||summary->writeFlags!=0u||summary->cpuFilterMoveCount!=0u||
     summary->totalCount>ulong(p.moveCount)*16ul)return;
  PairQueryRecord r=records[i];
  for(uint j=0u;j<r.count;++j){PairCandidate c=candidates[r.offset+r.count-1u-j];
    PairContactSeed seed;seed.shapeIndexA=c.shapeIndex;seed.shapeIndexB=r.queryShapeIndex;seeds[r.offset+j]=seed;}
}
float3 point_segment(float3 a,float3 b,float3 q){float3 ab=b-a;float alpha=dot(ab,q-a);
  if(alpha<=0.0f)return a;float denominator=dot(ab,ab);if(alpha>denominator)return b;return a+(alpha/denominator)*ab;}
SegmentResult segment_distance(float3 p1,float3 q1,float3 p2,float3 q2){SegmentResult r;float3 d1=q1-p1,d2=q2-p2,delta=p1-p2;
  float a=dot(d1,d1),b=dot(d1,d2),c=dot(d1,delta),e=dot(d2,d2),f=dot(d2,delta);float s1,s2;
  if(a<100.0f*1.19209290e-7f&&e<100.0f*1.19209290e-7f){s1=0.0f;s2=0.0f;}
  else if(a<100.0f*1.19209290e-7f){s1=0.0f;s2=clamp(f/e,0.0f,1.0f);}
  else if(e<100.0f*1.19209290e-7f){s1=clamp(-c/a,0.0f,1.0f);s2=0.0f;}
  else{float denom=a*e-b*b;s1=denom>1000.0f*1.17549435e-38f?clamp((b*f-c*e)/denom,0.0f,1.0f):0.0f;
    s2=(b*s1+f)/e;if(s2<0.0f){s1=clamp(-c/a,0.0f,1.0f);s2=0.0f;}else if(s2>1.0f){s1=clamp((b-c)/a,0.0f,1.0f);s2=1.0f;}}
  r.point1=p1+s1*d1;r.fraction1=s1;r.point2=p2+s2*d2;r.fraction2=s2;return r;}
uint clip_segment(thread float3& a,thread uint& fa,thread float3& b,thread uint& fb,float3 n,float planeOffset){
  float da=dot(n,a)-planeOffset,db=dot(n,b)-planeOffset;float3 oa=a,ob=b;uint ofa=fa,ofb=fb;uint count=0u;
  if(da<=0.0f){a=oa;fa=ofa;count=1u;}if(db<=0.0f){if(count==0u){a=ob;fa=ofb;}else{b=ob;fb=ofb;}count++;}
  if(da*db<0.0f){float t=da/(da-db);float3 v=(1.0f-t)*oa+t*ob;uint feature=da>0.0f?ofa:ofb;
    if(count==0u){a=v;fa=feature;}else{b=v;fb=feature;}count++;}return count;}
float3 closest_triangle(float3 p,float3 a,float3 b,float3 c){float3 ab=b-a,ac=c-a,ap=p-a;float d1=dot(ab,ap),d2=dot(ac,ap);
  if(d1<=0.0f&&d2<=0.0f)return a;float3 bp=p-b;float d3=dot(ab,bp),d4=dot(ac,bp);if(d3>=0.0f&&d4<=d3)return b;
  float vc=d1*d4-d3*d2;if(vc<=0.0f&&d1>=0.0f&&d3<=0.0f){float v=d1/(d1-d3);return a+v*ab;}
  float3 cp=p-c;float d5=dot(ab,cp),d6=dot(ac,cp);if(d6>=0.0f&&d5<=d6)return c;
  float vb=d5*d2-d1*d6;if(vb<=0.0f&&d2>=0.0f&&d6<=0.0f){float w=d2/(d2-d6);return a+w*ac;}
  float va=d3*d6-d5*d4;if(va<=0.0f&&(d4-d3)>=0.0f&&(d5-d6)>=0.0f){float w=(d4-d3)/((d4-d3)+(d5-d6));return b+w*(c-b);}
  float denom=1.0f/(va+vb+vc),v=vb*denom,w=vc*denom;return a+v*ab+w*ac;}
float3 box_cross(float3 a,float3 b){return float3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);}
float box_dot(float3 a,float3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
float3 box_rotate(float4 q,float3 v){float3 t1=box_cross(q.xyz,v);float3 t2=float3(t1.x+q.w*v.x,t1.y+q.w*v.y,t1.z+q.w*v.z);
  float3 t3=box_cross(q.xyz,t2);return float3(v.x+2.0f*t3.x,v.y+2.0f*t3.y,v.z+2.0f*t3.z);}
float3 box_inv_rotate(float4 q,float3 v){float3 t1=box_cross(q.xyz,v);float3 t2=float3(t1.x-q.w*v.x,t1.y-q.w*v.y,t1.z-q.w*v.z);
  float3 t3=box_cross(q.xyz,t2);return float3(v.x+2.0f*t3.x,v.y+2.0f*t3.y,v.z+2.0f*t3.z);}
float4 box_inv_mul_quat(float4 a,float4 b){float3 t1=box_cross(b.xyz,a.xyz);float3 t2=float3(t1.x+a.w*b.x,t1.y+a.w*b.y,t1.z+a.w*b.z);
  float3 t3=float3(t2.x-b.w*a.x,t2.y-b.w*a.y,t2.z-b.w*a.z);return float4(t3,a.w*b.w+box_dot(a.xyz,b.xyz));}
float3 box_clip_intersection(float3 a,float3 b,float da,float db){float denominator=da-db;float fraction=da/denominator;
  float3 delta=b-a;float3 scaled=fraction*delta;return a+scaled;}
uint box_make_feature(uint owner1,uint index1,uint owner2,uint index2){return (owner1<<24u)|(index1<<16u)|(owner2<<8u)|index2;}
uint box_flip_feature(uint feature){uint owner1=feature>>24u,index1=(feature>>16u)&255u,owner2=(feature>>8u)&255u,index2=feature&255u;
  return ((1u-owner2)<<24u)|(index2<<16u)|((1u-owner1)<<8u)|index1;}
float3 box_arbitrary_perp(float3 v){float3 r;if(fabs(v.x)>0.5f)r=float3(0.67f*v.y-0.42f*v.z,-0.67f*v.x,0.42f*v.x);
  else if(fabs(v.y)>0.5f)r=float3(0.67f*v.y,-0.67f*v.x-0.42f*v.z,0.42f*v.y);
  else r=float3(0.67f*v.z,-0.42f*v.z,-0.67f*v.x+0.42f*v.y);return normalize(r);}
uint box_find_incident_face(ShapeGeometry hull,float3 refNormal,uint vertexIndex,const device float4* hullPoints,
                            const device float4* hullPlanes,const device HullEdge* hullEdges){
  uint start=uint(hullPoints[hull.pointOffset+vertexIndex].w),edgeIndex=start,minEdge=0xffffffffu;
  float minProjection=3.40282347e+38f;float3 origin=hullPoints[hull.pointOffset+vertexIndex].xyz;
  for(uint iteration=0u;iteration<8u;++iteration){HullEdge edge=hullEdges[hull.edgeOffset+edgeIndex];
    HullEdge twin=hullEdges[hull.edgeOffset+edge.twin];float3 axis=normalize(hullPoints[hull.pointOffset+twin.origin].xyz-origin);
    float projection=fabs(dot(axis,refNormal));if(projection<minProjection){minProjection=projection;minEdge=edgeIndex;}
    edgeIndex=twin.next;if(edgeIndex==start)break;}
  if(minEdge==0xffffffffu)return 0xffffffffu;HullEdge edge=hullEdges[hull.edgeOffset+minEdge];
  HullEdge twin=hullEdges[hull.edgeOffset+edge.twin];float d1=dot(hullPlanes[hull.planeOffset+edge.face].xyz,refNormal);
  float d2=dot(hullPlanes[hull.planeOffset+twin.face].xyz,refNormal);return d1<d2?edge.face:twin.face;}
uint box_clip_polygon(thread BoxClipVertex* out,thread const BoxClipVertex* polygon,uint count,float3 normal,float offset,
                      uint edgeIndex,float3 refNormal,float refOffset){
  BoxClipVertex vertex1=polygon[count-1u];float distance1=dot(normal,vertex1.position)-offset;uint outCount=0u;
  for(uint index=0u;index<count;++index){BoxClipVertex vertex2=polygon[index];float distance2=dot(normal,vertex2.position)-offset;
    if(distance1<=0.0f&&distance2<=0.0f){if(outCount>=8u)return 9u;out[outCount++]=vertex2;}
    else if(distance1<=0.0f&&distance2>0.0f){BoxClipVertex clipped;
      clipped.position=box_clip_intersection(vertex1.position,vertex2.position,distance1,distance2);clipped.separation=dot(refNormal,clipped.position)-refOffset;
      clipped.feature=(vertex2.feature&0xffff0000u)|edgeIndex;if(outCount>=8u)return 9u;out[outCount++]=clipped;}
    else if(distance2<=0.0f&&distance1>0.0f){BoxClipVertex clipped;
      clipped.position=box_clip_intersection(vertex1.position,vertex2.position,distance1,distance2);clipped.separation=dot(refNormal,clipped.position)-refOffset;
      clipped.feature=(vertex1.feature&0x0000ffffu)|(edgeIndex<<16u);if(outCount>=8u)return 9u;out[outCount++]=clipped;
      if(outCount>=8u)return 9u;out[outCount++]=vertex2;}vertex1=vertex2;distance1=distance2;}return outCount;}
void box_reduce(thread BoxManifold& manifold,thread BoxClipVertex* points,uint count,float speculativeDistance){
  if(count<=4u){for(uint j=0u;j<count;++j)manifold.points[j]=points[j];manifold.pointCount=count;return;}
  float bias=0.95f,tolSqr=speculativeDistance*speculativeDistance;float3 search=box_arbitrary_perp(manifold.normal);
  int best=-1;float bestScore=-3.40282347e+38f;for(uint j=0u;j<count;++j){if(points[j].separation>speculativeDistance)continue;
    float score=-points[j].separation+dot(search,points[j].position);if(bias*score>bestScore){best=int(j);bestScore=score;}}
  if(best<0)return;manifold.points[0]=points[best];points[best]=points[count-1u];count-=1u;manifold.pointCount=1u;
  float3 a=manifold.points[0].position;best=-1;bestScore=0.0f;for(uint j=0u;j<count;++j){float3 d=points[j].position-a;
    float3 v=d-dot(d,manifold.normal)*manifold.normal;float separation=max(0.0f,-points[j].separation);
    float score=dot(v,v)+4.0f*separation*separation;if(bias*score>bestScore){best=int(j);bestScore=score;}}
  if(bestScore<tolSqr||best<0)return;manifold.points[1]=points[best];points[best]=points[count-1u];count-=1u;manifold.pointCount=2u;
  float3 b=manifold.points[1].position,ba=b-a;best=-1;bestScore=tolSqr;float bestSignedArea=0.0f;
  for(uint j=0u;j<count;++j){float signedArea=dot(manifold.normal,cross(ba,points[j].position-a));float score=fabs(signedArea);
    if(bias*score>=bestScore){best=int(j);bestScore=score;bestSignedArea=signedArea;}}if(best<0)return;
  manifold.points[2]=points[best];points[best]=points[count-1u];count-=1u;manifold.pointCount=3u;float3 c=manifold.points[2].position;
  best=-1;bestScore=tolSqr;float sign=bestSignedArea<0.0f?-1.0f:1.0f;for(uint j=0u;j<count;++j){float3 q=points[j].position;
    float u1=sign*dot(manifold.normal,cross(q-a,ba));float u2=sign*dot(manifold.normal,cross(q-b,c-b));
    float u3=sign*dot(manifold.normal,cross(q-c,a-c));float score=max(u1,max(u2,u3));if(bias*score>bestScore){best=int(j);bestScore=score;}}
  if(best>=0){manifold.points[3]=points[best];manifold.pointCount=4u;}}
void box_build_face(thread BoxManifold& manifold,ShapeGeometry refHull,ShapeGeometry incHull,float4 incRotation,float3 incPosition,
                    uint refFace,uint incVertex,float speculativeDistance,const device float4* hullPoints,
                    const device float4* hullPlanes,const device HullEdge* hullEdges,const device uint* hullFaces){
  manifold.valid=0u;manifold.pointCount=0u;float4 refPlane=hullPlanes[refHull.planeOffset+refFace];manifold.normal=refPlane.xyz;
  float3 refNormalInInc=inv_rotate(incRotation,refPlane.xyz);uint incFace=box_find_incident_face(incHull,refNormalInInc,incVertex,hullPoints,hullPlanes,hullEdges);
  if(incFace==0xffffffffu)return;thread BoxClipVertex buffer1[8];thread BoxClipVertex buffer2[8];uint start=hullFaces[incHull.planeOffset+incFace];
  uint edgeIndex=start,pointCount=0u;for(uint iteration=0u;iteration<8u;++iteration){HullEdge edge=hullEdges[incHull.edgeOffset+edgeIndex];
    uint nextIndex=edge.next;HullEdge next=hullEdges[incHull.edgeOffset+nextIndex];if(pointCount>=8u)return;BoxClipVertex clipped;
    clipped.position=rotate(incRotation,hullPoints[incHull.pointOffset+next.origin].xyz)+incPosition;
    clipped.separation=dot(refPlane.xyz,clipped.position)-refPlane.w;clipped.feature=box_make_feature(1u,edgeIndex,1u,nextIndex);
    buffer1[pointCount++]=clipped;edgeIndex=nextIndex;if(edgeIndex==start)break;}if(pointCount!=4u)return;
  thread BoxClipVertex* input=buffer1;thread BoxClipVertex* output=buffer2;start=hullFaces[refHull.planeOffset+refFace];edgeIndex=start;
  uint sideCount=0u;for(uint iteration=0u;iteration<8u;++iteration){HullEdge edge=hullEdges[refHull.edgeOffset+edgeIndex];
    uint nextIndex=edge.next;HullEdge next=hullEdges[refHull.edgeOffset+nextIndex];float3 vertex1=hullPoints[refHull.pointOffset+edge.origin].xyz;
    float3 vertex2=hullPoints[refHull.pointOffset+next.origin].xyz;float3 tangent=normalize(vertex2-vertex1);float3 binormal=cross(tangent,refPlane.xyz);
    pointCount=box_clip_polygon(output,input,pointCount,binormal,dot(binormal,vertex1),edgeIndex,refPlane.xyz,refPlane.w);
    if(pointCount<3u||pointCount>8u)return;thread BoxClipVertex* swap=input;input=output;output=swap;edgeIndex=nextIndex;sideCount+=1u;
    if(edgeIndex==start)break;}if(sideCount!=4u)return;float minSeparation=3.40282347e+38f;
  for(uint j=0u;j<pointCount;++j){minSeparation=min(minSeparation,input[j].separation);input[j].position-=0.5f*input[j].separation*refPlane.xyz;}
  manifold.minSeparation=minSeparation;manifold.valid=1u;if(minSeparation>=speculativeDistance)return;
  box_reduce(manifold,input,pointCount,speculativeDistance);}
BoxEdgeQuery box_query_edges(ShapeGeometry hullA,ShapeGeometry hullB,float4 rotationB,float3 positionB,
                             const device float4* hullPoints,const device float4* hullPlanes,const device HullEdge* hullEdges,
                             float speculativeDistance){
  BoxEdgeQuery query;query.normal=float3(0.0f);query.separation=-3.40282347e+38f;query.indexA=0u;query.indexB=0u;query.valid=0u;
  const float eps=-0.0001f,squaredTolerance=0.000025f;
  for(uint indexB=0u;indexB<hullB.edgeCount;indexB+=2u){HullEdge edgeB=hullEdges[hullB.edgeOffset+indexB];
    HullEdge twinB=hullEdges[hullB.edgeOffset+edgeB.twin];float3 pointB=hullPoints[hullB.pointOffset+edgeB.origin].xyz;
    float3 edgeVectorB=hullPoints[hullB.pointOffset+twinB.origin].xyz-pointB;float3 c=-rotate(rotationB,hullPlanes[hullB.planeOffset+edgeB.face].xyz);
    float3 d=-rotate(rotationB,hullPlanes[hullB.planeOffset+twinB.face].xyz),dc=-rotate(rotationB,edgeVectorB);
    float3 bv0=-(rotate(rotationB,pointB)+positionB);
    for(uint indexA=0u;indexA<hullA.edgeCount;indexA+=2u){HullEdge edgeA=hullEdges[hullA.edgeOffset+indexA];
      HullEdge twinA=hullEdges[hullA.edgeOffset+edgeA.twin];float3 av0=hullPoints[hullA.pointOffset+edgeA.origin].xyz;
      float3 edgeVectorA=hullPoints[hullA.pointOffset+twinA.origin].xyz-av0;float3 a=hullPlanes[hullA.planeOffset+edgeA.face].xyz;
      float3 b=hullPlanes[hullA.planeOffset+twinA.face].xyz;float cba=dot(c,edgeVectorA),dba=dot(d,edgeVectorA);
      if(cba*dba>=eps)continue;float adc=dot(a,dc),bdc=dot(b,dc);if(adc*bdc>=eps||cba*bdc>=eps)continue;
      if(max(cba*cba,dba*dba)<=squaredTolerance*dot(edgeVectorA,edgeVectorA))continue;float t=-cba/(dba-cba);
      float3 axis=c+t*(d-c);float axisLengthSquared=dot(axis,axis);if(axisLengthSquared<=1000.0f*1.17549435e-38f)continue;
      axis*=rsqrt(axisLengthSquared);float separation=-dot(axis,av0+bv0);
      if(separation>query.separation){query.normal=axis;query.separation=separation;query.indexA=indexA;query.indexB=indexB;query.valid=1u;
        if(separation>speculativeDistance)return query;}}}return query;}
uint box_build_edge(thread BoxManifold& manifold,ShapeGeometry hullA,ShapeGeometry hullB,float4 rotationB,float3 positionB,
                    BoxEdgeQuery query,const device float4* hullPoints,const device HullEdge* hullEdges){
  HullEdge edgeA=hullEdges[hullA.edgeOffset+query.indexA],twinA=hullEdges[hullA.edgeOffset+edgeA.twin];
  HullEdge edgeB=hullEdges[hullB.edgeOffset+query.indexB],twinB=hullEdges[hullB.edgeOffset+edgeB.twin];
  float3 pA=hullPoints[hullA.pointOffset+edgeA.origin].xyz,qA=hullPoints[hullA.pointOffset+twinA.origin].xyz,eA=qA-pA;
  float3 pB=rotate(rotationB,hullPoints[hullB.pointOffset+edgeB.origin].xyz)+positionB;
  float3 qB=rotate(rotationB,hullPoints[hullB.pointOffset+twinB.origin].xyz)+positionB,eB=qB-pB,w=pA-pB;
  float a11=dot(eA,eA),a12=-dot(eA,eB),a21=dot(eB,eA),a22=-dot(eB,eB);
  float b1=-dot(eA,w),b2=-dot(eB,w),det=a11*a22-a12*a21,s1,s2;
  if(det*det<1000.0f*1.17549435e-38f){s1=dot(pB-pA,eA)/a11;s2=0.0f;}
  else{s1=(a22*b1-a12*b2)/det;s2=(a11*b2-a21*b1)/det;}
  if(s1<0.0f||s1>1.0f||s2<0.0f||s2>1.0f)return 0u;float3 pointA=pA+s1*eA,pointB=pB+s2*eB;
  manifold.normal=query.normal;manifold.pointCount=1u;manifold.minSeparation=dot(query.normal,pointB-pointA);manifold.valid=1u;
  manifold.points[0].position=0.5f*(pointA+pointB);manifold.points[0].separation=manifold.minSeparation;
  manifold.points[0].feature=box_make_feature(0u,query.indexA,1u,query.indexB);return 1u;}
kernel void b3_contact_input_bootstrap(const device ContactInputSeed* seeds [[buffer(0)]],
                                        device ConvexManifoldInput* inputs [[buffer(1)]],
                                        const device ShapeGeometry* geometry [[buffer(2)]],
                                        const device BodyTransform* transforms [[buffer(3)]],
                                        device atomic_uint* status [[buffer(4)]],
                                        constant ContactInputBootstrapParams& p [[buffer(5)]],
                                        uint i [[thread_position_in_grid]]) {
  if(i>=p.count)return;ContactInputSeed seed=seeds[i];ConvexManifoldInput out={};inputs[i]=out;
  if(seed.contactId>=p.contactCapacity||seed.generation==0u||seed.shapeIdA>=p.shapeCount||seed.shapeIdB>=p.shapeCount){atomic_fetch_or_explicit(status,1u,memory_order_relaxed);return;}
  ShapeGeometry a=geometry[seed.shapeIdA],b=geometry[seed.shapeIdB];
  if(a.supported==0u||b.supported==0u||a.bodyId<0||b.bodyId<0||uint(a.bodyId)>=p.bodyCount||uint(b.bodyId)>=p.bodyCount){atomic_fetch_or_explicit(status,2u,memory_order_relaxed);return;}
  BodyTransform ta=transforms[a.bodyId],tb=transforms[b.bodyId];if(ta.supported==0u||tb.supported==0u){atomic_fetch_or_explicit(status,4u,memory_order_relaxed);return;}
  bool supported=(a.type==5u&&b.type==5u)||(a.type==0u&&b.type==5u)||(a.type==0u&&b.type==0u)||(a.type==3u&&b.type==5u)||
    (a.type==3u&&b.type==3u&&a.pointCount==8u&&a.planeCount==6u&&a.edgeCount==24u&&b.pointCount==8u&&b.planeCount==6u&&b.edgeCount==24u);
  if(!supported){atomic_fetch_or_explicit(status,8u,memory_order_relaxed);return;}
  out.eligible=1u;out.shapeIdA=seed.shapeIdA;out.shapeIdB=seed.shapeIdB;out.contactId=seed.contactId;out.contactGeneration=seed.generation;
  out.prepareEligible=p.prepareEligible;out.indexA=ta.index;out.indexB=tb.index;inputs[i]=out;
}
kernel void b3_pair_seed_input_bootstrap(const device PairContactSeed* seeds [[buffer(0)]],
                                             device ConvexManifoldInput* inputs [[buffer(1)]],
                                             const device ShapeGeometry* geometry [[buffer(2)]],
                                             const device BodyTransform* transforms [[buffer(3)]],
                                             device atomic_uint* status [[buffer(4)]],
                                             constant ContactInputBootstrapParams& p [[buffer(5)]],
                                             device atomic_uint* bodyOwners [[buffer(6)]],
                                             uint i [[thread_position_in_grid]]) {
  if(i>=p.count)return;PairContactSeed seed=seeds[i];ConvexManifoldInput out={};inputs[i]=out;
  if(seed.shapeIndexA<0||seed.shapeIndexB<0||uint(seed.shapeIndexA)>=p.shapeCount||uint(seed.shapeIndexB)>=p.shapeCount||i>=p.contactCapacity){atomic_fetch_or_explicit(status,1u,memory_order_relaxed);return;}
  uint shapeA=uint(seed.shapeIndexA),shapeB=uint(seed.shapeIndexB);ShapeGeometry a=geometry[shapeA],b=geometry[shapeB];
  if(a.type>b.type){uint shape=shapeA;shapeA=shapeB;shapeB=shape;ShapeGeometry g=a;a=b;b=g;}
  if(a.supported==0u||b.supported==0u||a.bodyId<0||b.bodyId<0||uint(a.bodyId)>=p.bodyCount||uint(b.bodyId)>=p.bodyCount){atomic_fetch_or_explicit(status,2u,memory_order_relaxed);return;}
  BodyTransform ta=transforms[a.bodyId],tb=transforms[b.bodyId];if(ta.supported==0u||tb.supported==0u){atomic_fetch_or_explicit(status,4u,memory_order_relaxed);return;}
  if(p.p0!=0u){bool staticA=ta.index<0,staticB=tb.index<0;if(staticA==staticB){atomic_fetch_or_explicit(status,16u,memory_order_relaxed);return;}uint dynamicBody=uint(staticA?b.bodyId:a.bodyId);if(atomic_exchange_explicit(bodyOwners+dynamicBody,i+1u,memory_order_relaxed)!=0u){atomic_fetch_or_explicit(status,32u,memory_order_relaxed);return;}}
  bool supported=(a.type==5u&&b.type==5u)||(a.type==0u&&b.type==5u)||(a.type==0u&&b.type==0u)||(a.type==3u&&b.type==5u)||
    (a.type==3u&&b.type==3u&&a.pointCount==8u&&a.planeCount==6u&&a.edgeCount==24u&&b.pointCount==8u&&b.planeCount==6u&&b.edgeCount==24u);
  if(!supported){atomic_fetch_or_explicit(status,8u,memory_order_relaxed);return;}
  out.eligible=1u;out.shapeIdA=shapeA;out.shapeIdB=shapeB;out.contactId=i;out.contactGeneration=1u;
  out.prepareEligible=p.prepareEligible;out.indexA=ta.index;out.indexB=tb.index;inputs[i]=out;
}
kernel void b3_convex_manifolds(const device ConvexManifoldInput* inputs [[buffer(0)]],
                                device ConvexManifoldResult* results [[buffer(1)]],
                                constant ConvexManifoldParams& p [[buffer(2)]],
                                const device float4* hullPoints [[buffer(3)]],
                                const device float4* hullPlanes [[buffer(4)]],
                                const device HullTriangle* hullTriangles [[buffer(5)]],
                                const device ShapeGeometry* shapeGeometry [[buffer(6)]],
                                const device BodyTransform* bodyTransforms [[buffer(7)]],
                                const device HullEdge* hullEdges [[buffer(8)]],
                                const device uint* hullFaces [[buffer(9)]],
                                const device ConvexManifoldResult* previousTable [[buffer(10)]],
                                uint i [[thread_position_in_grid]]) {
  if(i>=p.contactCount)return; ConvexManifoldInput in=inputs[i]; ConvexManifoldResult out={};out.inputIndex=i;
  if(in.contactId<p.previousTableCount){ConvexManifoldResult previous=previousTable[in.contactId];if(previous.eligible!=0u&&previous.contactId==in.contactId&&previous.contactGeneration==in.contactGeneration){in.satSeparation=previous.satSeparation;in.satCache=previous.satCache;}}
  if(in.eligible==0u){results[i]=out;return;} out.eligible=1u;
  ShapeGeometry geometryA=shapeGeometry[in.shapeIdA],geometryB=shapeGeometry[in.shapeIdB];
  if(geometryA.supported==0u||geometryB.supported==0u||geometryA.bodyId<0||geometryB.bodyId<0||
     uint(geometryA.bodyId)>=p.bodyCount||uint(geometryB.bodyId)>=p.bodyCount){out.eligible=0u;results[i]=out;return;}
  BodyTransform transformA=bodyTransforms[geometryA.bodyId],transformB=bodyTransforms[geometryB.bodyId];
  if(transformA.supported==0u||transformB.supported==0u){out.eligible=0u;results[i]=out;return;}float3 d;
  if(((transformA.flags|transformB.flags)&0x40u)!=0u)out.residentFlags|=4u;
#if defined(B3_DOUBLE_PRECISION)
  d=float3(b3_vf64_difference(transformB.pxBits,transformA.pxBits),
           b3_vf64_difference(transformB.pyBits,transformA.pyBits),
           b3_vf64_difference(transformB.pzBits,transformA.pzBits));
#else
  d=float3(transformB.px-transformA.px,transformB.py-transformA.py,transformB.pz-transformA.pz);
#endif
  float4 qA=float4(transformA.qx,transformA.qy,transformA.qz,transformA.qw);
  float4 qB=float4(transformB.qx,transformB.qy,transformB.qz,transformB.qw);
  float3 relativePosition=box_inv_rotate(qA,d); float4 relativeRotation=box_inv_mul_quat(qA,qB);
  float3 a1=float3(geometryA.point1X,geometryA.point1Y,geometryA.point1Z),a2=float3(geometryA.point2X,geometryA.point2Y,geometryA.point2Z);
  float3 b1=rotate(relativeRotation,float3(geometryB.point1X,geometryB.point1Y,geometryB.point1Z))+relativePosition;
  float3 b2=rotate(relativeRotation,float3(geometryB.point2X,geometryB.point2Y,geometryB.point2Z))+relativePosition;
  if(geometryA.type==3u&&geometryB.type==3u){
    if(geometryA.pointCount!=8u||geometryA.planeCount!=6u||geometryA.edgeCount!=24u||
       geometryB.pointCount!=8u||geometryB.planeCount!=6u||geometryB.edgeCount!=24u){out.eligible=0u;results[i]=out;return;}
    float3 lowerA=float3(3.40282347e+38f),upperA=float3(-3.40282347e+38f),lowerB=lowerA,upperB=upperA;
    for(uint pointIndex=0u;pointIndex<8u;++pointIndex){float3 pointA=hullPoints[geometryA.pointOffset+pointIndex].xyz;float3 pointB=hullPoints[geometryB.pointOffset+pointIndex].xyz;
      lowerA=min(lowerA,pointA);upperA=max(upperA,pointA);lowerB=min(lowerB,pointB);upperB=max(upperB,pointB);}
    float3 extentA=upperA-lowerA,extentB=upperB-lowerB;float minA=min(extentA.x,min(extentA.y,extentA.z));
    float minB=min(extentB.x,min(extentB.y,extentB.z)),maxA=max(extentA.x,max(extentA.y,extentA.z)),maxB=max(extentB.x,max(extentB.y,extentB.z));
    if(minA<=p.linearSlop||minB<=p.linearSlop||maxA>16.0f*minA||maxB>16.0f*minB){out.eligible=0u;results[i]=out;return;}
    BoxManifold manifold={};uint cacheType=in.satCache&255u,cacheIndexA=(in.satCache>>8u)&255u,cacheIndexB=(in.satCache>>16u)&255u;uint usedCache=0u;
    if(cacheType==2u&&cacheIndexA<geometryA.planeCount){float4 plane=hullPlanes[geometryA.planeOffset+cacheIndexA];float3 direction=-box_inv_rotate(qB,box_rotate(qA,plane.xyz));
      float bestDot=-3.40282347e+38f;uint support=0u;for(uint pointIndex=0u;pointIndex<geometryB.pointCount;++pointIndex){float value=dot(direction,hullPoints[geometryB.pointOffset+pointIndex].xyz);
        if(value>bestDot){bestDot=value;support=pointIndex;}}float3 supportPoint=box_rotate(relativeRotation,hullPoints[geometryB.pointOffset+support].xyz)+relativePosition;
      float separation=dot(plane.xyz,supportPoint)-plane.w;
      if(separation>=p.speculativeDistance){out.satSeparation=in.satSeparation;out.satCache=(in.satCache&0x00ffffffu)|0x01000000u;results[i]=out;return;}
      box_build_face(manifold,geometryA,geometryB,relativeRotation,relativePosition,cacheIndexA,support,p.speculativeDistance,hullPoints,hullPlanes,hullEdges,hullFaces);
      if(manifold.valid!=0u&&manifold.pointCount>0u&&fabs(in.satSeparation-manifold.minSeparation)<p.linearSlop){usedCache=1u;out.satSeparation=in.satSeparation;out.satCache=(in.satCache&0x00ffffffu)|0x01000000u;}}
    else if(cacheType==3u&&cacheIndexB<geometryB.planeCount){float4 plane=hullPlanes[geometryB.planeOffset+cacheIndexB];float3 direction=-box_inv_rotate(qA,box_rotate(qB,plane.xyz));
      float bestDot=-3.40282347e+38f;uint support=0u;for(uint pointIndex=0u;pointIndex<geometryA.pointCount;++pointIndex){float value=dot(direction,hullPoints[geometryA.pointOffset+pointIndex].xyz);
        if(value>bestDot){bestDot=value;support=pointIndex;}}float3 supportPoint=box_inv_rotate(relativeRotation,hullPoints[geometryA.pointOffset+support].xyz-relativePosition);
      float separation=dot(plane.xyz,supportPoint)-plane.w;
      if(separation>=p.speculativeDistance){out.satSeparation=in.satSeparation;out.satCache=(in.satCache&0x00ffffffu)|0x01000000u;results[i]=out;return;}
      float4 inverseRotation=float4(-relativeRotation.xyz,relativeRotation.w);float3 inversePosition=box_inv_rotate(relativeRotation,-relativePosition);
      box_build_face(manifold,geometryB,geometryA,inverseRotation,inversePosition,cacheIndexB,support,p.speculativeDistance,hullPoints,hullPlanes,hullEdges,hullFaces);
      if(manifold.valid!=0u){manifold.normal=-box_rotate(relativeRotation,manifold.normal);for(uint j=0u;j<manifold.pointCount;++j){manifold.points[j].position=box_rotate(relativeRotation,manifold.points[j].position)+relativePosition;manifold.points[j].feature=box_flip_feature(manifold.points[j].feature);}}
      if(manifold.valid!=0u&&manifold.pointCount>0u&&fabs(in.satSeparation-manifold.minSeparation)<p.linearSlop){usedCache=1u;out.satSeparation=in.satSeparation;out.satCache=(in.satCache&0x00ffffffu)|0x01000000u;}}
    if(usedCache==0u){manifold=BoxManifold{};out.satSeparation=0.0f;out.satCache=0u;
    float faceASeparation=-3.40282347e+38f,faceBSeparation=-3.40282347e+38f;uint faceA=0u,vertexB=0u,faceB=0u,vertexA=0u;
    for(uint face=0u;face<geometryA.planeCount;++face){float4 plane=hullPlanes[geometryA.planeOffset+face];float3 direction=-box_inv_rotate(qB,box_rotate(qA,plane.xyz));
      float planeSeparation=dot(plane.xyz,relativePosition)-plane.w,supportValue=-3.40282347e+38f;uint support=0u;
      for(uint pointIndex=0u;pointIndex<geometryB.pointCount;++pointIndex){float value=dot(direction,hullPoints[geometryB.pointOffset+pointIndex].xyz);
        if(value>supportValue){supportValue=value;support=pointIndex;}}float separation=planeSeparation-supportValue;
      if(separation>faceASeparation){faceASeparation=separation;faceA=face;vertexB=support;}}
    for(uint face=0u;face<geometryB.planeCount;++face){float4 plane=hullPlanes[geometryB.planeOffset+face];float3 direction=-box_inv_rotate(qA,box_rotate(qB,plane.xyz));
      float planeSeparation=dot(direction,relativePosition)-plane.w,supportValue=-3.40282347e+38f;uint support=0u;
      for(uint pointIndex=0u;pointIndex<geometryA.pointCount;++pointIndex){float value=dot(direction,hullPoints[geometryA.pointOffset+pointIndex].xyz);
        if(value>supportValue){supportValue=value;support=pointIndex;}}float separation=planeSeparation-supportValue;
      if(separation>faceBSeparation){faceBSeparation=separation;faceB=face;vertexA=support;}}
    if(faceASeparation>p.speculativeDistance){out.satSeparation=faceASeparation;out.satCache=2u|(faceA<<8u)|(vertexB<<16u);results[i]=out;return;}
    if(faceBSeparation>p.speculativeDistance){out.satSeparation=faceBSeparation;out.satCache=3u|(vertexA<<8u)|(faceB<<16u);results[i]=out;return;}
    BoxEdgeQuery edgeQuery=box_query_edges(geometryA,geometryB,relativeRotation,relativePosition,hullPoints,hullPlanes,hullEdges,p.speculativeDistance);
    if(edgeQuery.valid!=0u&&edgeQuery.separation>p.speculativeDistance){out.satSeparation=edgeQuery.separation;out.satCache=4u|(edgeQuery.indexA<<8u)|(edgeQuery.indexB<<16u);results[i]=out;return;}
    if(faceASeparation>faceBSeparation){box_build_face(manifold,geometryA,geometryB,relativeRotation,relativePosition,faceA,vertexB,p.speculativeDistance,hullPoints,hullPlanes,hullEdges,hullFaces);
      if(manifold.valid!=0u&&manifold.pointCount>0u){out.satSeparation=manifold.minSeparation;out.satCache=2u|(faceA<<8u)|(vertexB<<16u);}}
    else{float4 inverseRotation=float4(-relativeRotation.xyz,relativeRotation.w);float3 inversePosition=box_inv_rotate(relativeRotation,-relativePosition);
      box_build_face(manifold,geometryB,geometryA,inverseRotation,inversePosition,faceB,vertexA,p.speculativeDistance,hullPoints,hullPlanes,hullEdges,hullFaces);
      if(manifold.valid!=0u){manifold.normal=-box_rotate(relativeRotation,manifold.normal);for(uint j=0u;j<manifold.pointCount;++j){
        manifold.points[j].position=box_rotate(relativeRotation,manifold.points[j].position)+relativePosition;manifold.points[j].feature=box_flip_feature(manifold.points[j].feature);}
        if(manifold.pointCount>0u){out.satSeparation=manifold.minSeparation;out.satCache=3u|(vertexA<<8u)|(faceB<<16u);}}}
    if(edgeQuery.valid!=0u&&(manifold.pointCount==0u||edgeQuery.separation>manifold.minSeparation+p.linearSlop)){
      BoxManifold edgeManifold={};if(box_build_edge(edgeManifold,geometryA,geometryB,relativeRotation,relativePosition,edgeQuery,hullPoints,hullEdges)!=0u){manifold=edgeManifold;out.satSeparation=edgeManifold.minSeparation;out.satCache=4u|(edgeQuery.indexA<<8u)|(edgeQuery.indexB<<16u);}}
    }
    if(manifold.valid==0u){out.eligible=0u;results[i]=out;return;}
    if(manifold.pointCount==0u){results[i]=out;return;}out.touching=1u;out.pointCount=manifold.pointCount;
    out.nx=manifold.normal.x;out.ny=manifold.normal.y;out.nz=manifold.normal.z;BoxClipVertex point=manifold.points[0];
    out.p1x=point.position.x;out.p1y=point.position.y;out.p1z=point.position.z;out.separation1=point.separation;out.feature1=point.feature;
    if(manifold.pointCount>1u){point=manifold.points[1];out.p2x=point.position.x;out.p2y=point.position.y;out.p2z=point.position.z;out.separation2=point.separation;out.feature2=point.feature;}
    if(manifold.pointCount>2u){point=manifold.points[2];out.p3x=point.position.x;out.p3y=point.position.y;out.p3z=point.position.z;out.separation3=point.separation;out.feature3=point.feature;}
    if(manifold.pointCount>3u){point=manifold.points[3];out.p4x=point.position.x;out.p4y=point.position.y;out.p4z=point.position.z;out.separation4=point.separation;out.feature4=point.feature;}
    results[i]=out;return;}
  if(geometryA.type==3u){ShapeGeometry hull=geometryA;
    float bestPlane=-3.40282347e+38f;float3 bestNormal=float3(0.0f,1.0f,0.0f);uint bestFace=0u;
    for(uint j=0u;j<hull.planeCount;++j){float4 plane=hullPlanes[hull.planeOffset+j];float separation=dot(plane.xyz,b1)-plane.w;
      if(separation>bestPlane){bestPlane=separation;bestNormal=plane.xyz;bestFace=j;}}
    float distance=0.0f;float3 closest=b1;float3 normal=bestNormal;
    if(bestPlane>0.0f){float bestDistanceSquared=3.40282347e+38f;float3 projection=b1-bestPlane*bestNormal;float faceDistanceSquared=3.40282347e+38f;
      for(uint j=0u;j<hull.triangleCount;++j){HullTriangle tri=hullTriangles[hull.triangleOffset+j];if(tri.face!=bestFace)continue;
        float3 q=closest_triangle(projection,hullPoints[hull.pointOffset+tri.index1].xyz,hullPoints[hull.pointOffset+tri.index2].xyz,hullPoints[hull.pointOffset+tri.index3].xyz);
        faceDistanceSquared=min(faceDistanceSquared,dot(projection-q,projection-q));}
      float faceTolerance=0.01f*p.linearSlop;if(faceDistanceSquared<=faceTolerance*faceTolerance){closest=projection;bestDistanceSquared=bestPlane*bestPlane;}else
      for(uint j=0u;j<hull.triangleCount;++j){HullTriangle tri=hullTriangles[hull.triangleOffset+j];
        float3 q=closest_triangle(b1,hullPoints[hull.pointOffset+tri.index1].xyz,hullPoints[hull.pointOffset+tri.index2].xyz,hullPoints[hull.pointOffset+tri.index3].xyz);
        float distanceSquared=dot(b1-q,b1-q);if(distanceSquared<bestDistanceSquared){bestDistanceSquared=distanceSquared;closest=q;}}
      distance=sqrt(bestDistanceSquared);if(distance>geometryB.radius+p.speculativeDistance){results[i]=out;return;}
      if(distance>geometryB.radius){out.eligible=0u;results[i]=out;return;}
      if(distance*distance>1000.0f*1.17549435e-38f)normal=(b1-closest)/distance;
    }else{distance=0.0f;closest=b1;}
    float separation=bestPlane<=0.0f?bestPlane-geometryB.radius:distance-geometryB.radius;float3 point=0.5f*(closest+b1-geometryB.radius*normal);
    out.touching=1u;out.pointCount=1u;out.nx=normal.x;out.ny=normal.y;out.nz=normal.z;out.p1x=point.x;out.p1y=point.y;out.p1z=point.z;
    out.separation1=separation;out.feature1=0u;results[i]=out;return;}
  float radius=geometryA.radius+geometryB.radius;float3 cpA,cpB;
  if(geometryA.type==5u){cpA=a1;cpB=b1;}else if(geometryB.type==5u){cpB=b1;cpA=point_segment(a1,a2,cpB);}
  else{SegmentResult sr=segment_distance(a1,a2,b1,b2);cpA=sr.point1;cpB=sr.point2;float3 initialOffset=cpB-cpA;
    float distanceSquared=dot(initialOffset,initialOffset),maxDistance=radius+p.speculativeDistance,minDistance=0.01f*p.linearSlop;
    if(distanceSquared>maxDistance*maxDistance||distanceSquared<minDistance*minDistance){results[i]=out;return;}
    float3 segmentA=a2-a1,segmentB=b2-b1;float lengthA=length(segmentA),lengthB=length(segmentB);
    if(lengthA<p.linearSlop||lengthB<p.linearSlop){results[i]=out;return;}float3 edgeA=segmentA/lengthA,edgeB=segmentB/lengthB;
    if(dot(cross(edgeA,edgeB),cross(edgeA,edgeB))<0.0025f){float3 v1=b1,v2=b2;uint f1=0u,f2=0x00010001u;
      uint count=clip_segment(v1,f1,v2,f2,-edgeA,-dot(edgeA,a1));if(count==2u)count=clip_segment(v1,f1,v2,f2,edgeA,dot(edgeA,a2));
      if(count==2u){float3 c1=point_segment(a1,a2,v1),c2=point_segment(a1,a2,v2);float d1=distance(c1,v1),d2=distance(c2,v2);
        if(d1<=radius&&d2<=radius){if(d1<minDistance||d2<minDistance){results[i]=out;return;}float3 n1=(v1-c1)/d1,n2=(v2-c2)/d2;
          float3 normal=normalize(n1+n2);float3 point1=0.5f*((v1+geometryA.radius*n1+c1)-geometryB.radius*normal);
          float3 point2=0.5f*((v2+geometryA.radius*n2+c2)-geometryB.radius*normal);out.touching=1u;out.pointCount=2u;
          out.nx=normal.x;out.ny=normal.y;out.nz=normal.z;out.p1x=point1.x;out.p1y=point1.y;out.p1z=point1.z;out.separation1=d1-radius;
          out.p2x=point2.x;out.p2y=point2.y;out.p2z=point2.z;out.separation2=d2-radius;out.feature1=f1;out.feature2=f2;results[i]=out;return;}}}}
  float3 offset=cpB-cpA;float distanceSq=dot(offset,offset);if(geometryB.type==5u&&distanceSq>radius*radius){results[i]=out;return;}
  float distance=sqrt(distanceSq);float3 normal=float3(0.0f,1.0f,0.0f);if(distance*distance>1000.0f*1.17549435e-38f)normal=offset/distance;
  float3 point=0.5f*((cpA+geometryA.radius*normal+cpB)-geometryB.radius*normal);out.touching=1u;out.pointCount=1u;
  out.nx=normal.x;out.ny=normal.y;out.nz=normal.z;out.p1x=point.x;out.p1y=point.y;out.p1z=point.z;
  out.separation1=distance-radius;out.feature1=0u;results[i]=out;
}
inline uint b3_manifold_stable(const ConvexManifoldResult r,const ConvexManifoldInput in,const device ImpulseResult* previous,const device PrepareInput* prepareTable,constant ManifoldCompactParams& p){
  if(p.p0==0u||r.eligible==0u||r.touching==0u||r.pointCount==0u||r.pointCount>4u||(r.residentFlags&4u)!=0u||(in.prepareEligible&2u)==0u||in.contactId>=p.previousCount)return 0u;
  ImpulseResult old=previous[in.contactId];if(old.contactId!=in.contactId||old.generation!=p.previousGeneration||old.contactGeneration!=in.contactGeneration||old.pointCount==0u||old.pointCount>4u)return 0u;
  PrepareInput prep=prepareTable[in.contactId];return prep.contactId==in.contactId&&prep.contactGeneration==in.contactGeneration&&prep.generation!=0u; }
inline uint b3_manifold_matches(const ConvexManifoldResult r,const ConvexManifoldInput in,const device ImpulseResult* previous){ImpulseResult old=previous[in.contactId];uint claimed=0u,matches=0u;
  for(uint pointIndex=0u;pointIndex<r.pointCount;++pointIndex){uint feature=pointIndex==0u?r.feature1:(pointIndex==1u?r.feature2:(pointIndex==2u?r.feature3:r.feature4));for(uint oldIndex=0u;oldIndex<old.pointCount;++oldIndex){uint bit=1u<<oldIndex;if((claimed&bit)==0u&&feature==old.points[oldIndex].featureId){claimed|=bit;matches+=1u;break;}}}return matches;}
inline uint b3_manifold_cold_transition(const ConvexManifoldResult r,const ConvexManifoldInput in,constant ManifoldCompactParams& p){return p.p1!=0u&&(in.prepareEligible&1u)!=0u&&r.eligible!=0u&&r.touching!=0u&&r.pointCount>0u&&r.pointCount<=4u&&(r.residentFlags&4u)==0u;}
kernel void b3_manifold_scan_blocks(device ConvexManifoldResult* results [[buffer(0)]],device ManifoldBlock* blocks [[buffer(1)]],
  const device ConvexManifoldInput* inputs [[buffer(2)]],const device ImpulseResult* previous [[buffer(3)]],
  const device PrepareInput* prepareTable [[buffer(4)]],constant ManifoldCompactParams& p [[buffer(5)]],uint i [[thread_position_in_grid]],uint ti [[thread_index_in_threadgroup]],
  uint group [[threadgroup_position_in_grid]],ushort lane [[thread_index_in_simdgroup]],ushort subgroup [[simdgroup_index_in_threadgroup]],
  ushort simdWidth [[threads_per_simdgroup]]){threadgroup uint totals[32],transitionTotals[32],stableTotals[32],silentTotals[32],matchTotals[32],offsets[32],transitionOffsets[32];
  uint stable=i<p.contactCount?b3_manifold_stable(results[i],inputs[i],previous,prepareTable,p):0u;
  uint transition=i<p.contactCount?b3_manifold_cold_transition(results[i],inputs[i],p):0u,silent=0u;
  uint exception=i<p.contactCount?(p.p1!=0u?1u-transition:(p.p0!=0u?1u-stable:results[i].eligible)):0u;uint matches=stable!=0u?b3_manifold_matches(results[i],inputs[i],previous):0u;
  uint local=simd_prefix_exclusive_sum(exception),transitionLocal=simd_prefix_exclusive_sum(transition);uint total=simd_sum(exception),transitionTotal=simd_sum(transition),stableTotal=simd_sum(stable),silentTotal=simd_sum(silent),matchTotal=simd_sum(matches);
  if(lane==0){totals[subgroup]=total;transitionTotals[subgroup]=transitionTotal;stableTotals[subgroup]=stableTotal;silentTotals[subgroup]=silentTotal;matchTotals[subgroup]=matchTotal;}threadgroup_barrier(mem_flags::mem_threadgroup);if(ti==0u){uint running=0u,transitionRunning=0u,stableRunning=0u,silentRunning=0u,matchRunning=0u;
    for(uint s=0u;s<256u/uint(simdWidth);++s){offsets[s]=running;transitionOffsets[s]=transitionRunning;running+=totals[s];transitionRunning+=transitionTotals[s];stableRunning+=stableTotals[s];silentRunning+=silentTotals[s];matchRunning+=matchTotals[s];}blocks[group]=ManifoldBlock{running,transitionRunning,stableRunning,silentRunning,0u,0u,matchRunning,0u};}
  threadgroup_barrier(mem_flags::mem_threadgroup);if(i<p.contactCount){ConvexManifoldResult r=results[i];
    r.scanOffset=local+offsets[subgroup];r.contactId=transitionLocal+transitionOffsets[subgroup];results[i]=r;}}
kernel void b3_manifold_prefix(device ManifoldBlock* blocks [[buffer(0)]],device ManifoldSummary* summary [[buffer(1)]],
  constant ManifoldCompactParams& p [[buffer(2)]]){uint exceptions=0u,transitions=0u,stable=0u,silent=0u,matches=0u,errors=0u;for(uint i=0u;i<p.blockCount;++i){ManifoldBlock b=blocks[i];
    b.exceptionOffset=exceptions;b.transitionOffset=transitions;blocks[i]=b;exceptions+=b.exceptionCount;transitions+=b.transitionCount;stable+=b.stableCount;silent+=b.silentCount;matches+=b.persistenceMatches;errors|=b.errorFlags;}summary->exceptionCount=ulong(exceptions);summary->transitionCount=transitions;summary->stableCount=stable;summary->silentCount=silent;summary->errorFlags=errors;summary->persistenceMatches=ulong(matches);}
kernel void b3_manifold_scatter(const device ConvexManifoldResult* results [[buffer(0)]],const device ManifoldBlock* blocks [[buffer(1)]],
  device ConvexManifoldResult* compact [[buffer(2)]],device ConvexManifoldInput* inputs [[buffer(3)]],
  const device ShapeGeometry* shapeGeometry [[buffer(4)]],const device BodyTransform* bodyTransforms [[buffer(5)]],
  device ConvexManifoldResult* table [[buffer(6)]],const device ImpulseResult* previous [[buffer(7)]],
  device PrepareInput* prepareTable [[buffer(8)]],device uint* transitions [[buffer(9)]],constant ManifoldCompactParams& p [[buffer(10)]],uint i [[thread_position_in_grid]]){
  if(i>=p.contactCount)return;ConvexManifoldResult r=results[i];uint transitionLocal=r.contactId;uint contactId=inputs[i].contactId;if(p.p1==1u){if(contactId>=p.p2)return;transitions[2u*contactId]=0u;transitions[2u*contactId+1u]=0u;}inputs[i].satSeparation=r.satSeparation;inputs[i].satCache=r.satCache;if(r.eligible==0u){if(p.p0!=0u||p.p1!=0u){uint output=blocks[i/256u].exceptionOffset+r.scanOffset;r.inputIndex=i;r.scanOffset=0u;r.contactId=contactId;compact[output]=r;}return;}ShapeGeometry geometryA=shapeGeometry[inputs[i].shapeIdA];
  ShapeGeometry geometryB=shapeGeometry[inputs[i].shapeIdB];BodyTransform transformA=bodyTransforms[geometryA.bodyId];
  BodyTransform transformB=bodyTransforms[geometryB.bodyId];float4 q=float4(transformA.qx,transformA.qy,transformA.qz,transformA.qw);
  float4 qb=float4(transformB.qx,transformB.qy,transformB.qz,transformB.qw);r.friction=sqrt(geometryA.friction*geometryB.friction);
  r.restitution=max(geometryA.restitution,geometryB.restitution);r.rollingResistance=max(geometryA.rollingResistance,geometryB.rollingResistance)*max(geometryA.rollingRadius,geometryB.rollingRadius);
  float3 tangentA=rotate(q,float3(geometryA.tangentVelocityX,geometryA.tangentVelocityY,geometryA.tangentVelocityZ));
  float3 tangentB=rotate(qb,float3(geometryB.tangentVelocityX,geometryB.tangentVelocityY,geometryB.tangentVelocityZ));float3 tangent=tangentA-tangentB;
  r.tangentVelocityX=tangent.x;r.tangentVelocityY=tangent.y;r.tangentVelocityZ=tangent.z;
  float3 centerA=rotate(q,float3(transformA.localCenterX,transformA.localCenterY,transformA.localCenterZ));
  float3 centerB=rotate(qb,float3(transformB.localCenterX,transformB.localCenterY,transformB.localCenterZ));float3 d;
#if defined(B3_DOUBLE_PRECISION)
  d=float3(b3_vf64_difference(transformB.pxBits,transformA.pxBits),b3_vf64_difference(transformB.pyBits,transformA.pyBits),b3_vf64_difference(transformB.pzBits,transformA.pzBits));
#else
  d=float3(transformB.px-transformA.px,transformB.py-transformA.py,transformB.pz-transformA.pz);
#endif
  if(r.pointCount>0u){float3 n=rotate(q,float3(r.nx,r.ny,r.nz));float3 p1=rotate(q,float3(r.p1x,r.p1y,r.p1z));float3 a=p1-centerA,b=p1-d-centerB;
    r.nx=n.x;r.ny=n.y;r.nz=n.z;r.p1x=a.x;r.p1y=a.y;r.p1z=a.z;r.anchorB1X=b.x;r.anchorB1Y=b.y;r.anchorB1Z=b.z;}
  if(r.pointCount>1u){float3 p2=rotate(q,float3(r.p2x,r.p2y,r.p2z));float3 a=p2-centerA,b=p2-d-centerB;
    r.p2x=a.x;r.p2y=a.y;r.p2z=a.z;r.anchorB2X=b.x;r.anchorB2Y=b.y;r.anchorB2Z=b.z;}
  if(r.pointCount>2u){float3 p3=rotate(q,float3(r.p3x,r.p3y,r.p3z));float3 a=p3-centerA,b=p3-d-centerB;
    r.p3x=a.x;r.p3y=a.y;r.p3z=a.z;r.anchorB3X=b.x;r.anchorB3Y=b.y;r.anchorB3Z=b.z;}
  if(r.pointCount>3u){float3 p4=rotate(q,float3(r.p4x,r.p4y,r.p4z));float3 a=p4-centerA,b=p4-d-centerB;
    r.p4x=a.x;r.p4y=a.y;r.p4z=a.z;r.anchorB4X=b.x;r.anchorB4Y=b.y;r.anchorB4Z=b.z;}
  r.contactGeneration=inputs[i].contactGeneration;ImpulseResult old={};uint oldValid=0u;if(r.pointCount>0u&&contactId<p.previousCount){old=previous[contactId];
    if(old.contactId==contactId&&old.generation==p.previousGeneration&&old.contactGeneration==inputs[i].contactGeneration&&old.pointCount>0u&&old.pointCount<=4u){
      oldValid=1u;r.residentFlags|=1u;uint claimed=0u;for(uint pointIndex=0u;pointIndex<r.pointCount;++pointIndex){uint feature=pointIndex==0u?r.feature1:(pointIndex==1u?r.feature2:(pointIndex==2u?r.feature3:r.feature4));
        for(uint oldIndex=0u;oldIndex<old.pointCount;++oldIndex){uint bit=1u<<oldIndex;if((claimed&bit)==0u&&feature==old.points[oldIndex].featureId){
          if(pointIndex==0u)r.normalImpulse1=old.points[oldIndex].normalImpulse;else if(pointIndex==1u)r.normalImpulse2=old.points[oldIndex].normalImpulse;else if(pointIndex==2u)r.normalImpulse3=old.points[oldIndex].normalImpulse;else r.normalImpulse4=old.points[oldIndex].normalImpulse;
          r.persistedBits|=1u<<pointIndex;claimed|=bit;break;}}}}}
  if(oldValid!=0u&&(inputs[i].prepareEligible&1u)!=0u&&r.touching!=0u&&r.pointCount>0u&&r.pointCount<=4u){PrepareInput prep=prepareTable[contactId];
    if(prep.contactId==contactId&&prep.contactGeneration==inputs[i].contactGeneration&&prep.generation!=0u){prep.indexA=transformA.index;prep.indexB=transformB.index;prep.generation=p.currentGeneration;
      prep.friction=r.friction;prep.restitution=r.restitution;prep.rollingResistance=r.rollingResistance;prep.tangentVelocityX=r.tangentVelocityX;prep.tangentVelocityY=r.tangentVelocityY;prep.tangentVelocityZ=r.tangentVelocityZ;
      prep.twistImpulse=old.twistImpulse;prep.frictionImpulseX=old.frictionX;prep.frictionImpulseY=old.frictionY;prep.frictionImpulseZ=old.frictionZ;
      prep.rollingImpulseX=old.rollingX;prep.rollingImpulseY=old.rollingY;prep.rollingImpulseZ=old.rollingZ;
      prep.points[0]=PreparePoint{r.p1x,r.p1y,r.p1z,r.separation1,r.anchorB1X,r.anchorB1Y,r.anchorB1Z,r.normalImpulse1,r.feature1};
      prep.points[1]=PreparePoint{r.p2x,r.p2y,r.p2z,r.separation2,r.anchorB2X,r.anchorB2Y,r.anchorB2Z,r.normalImpulse2,r.feature2};
      prep.points[2]=PreparePoint{r.p3x,r.p3y,r.p3z,r.separation3,r.anchorB3X,r.anchorB3Y,r.anchorB3Z,r.normalImpulse3,r.feature3};
      prep.points[3]=PreparePoint{r.p4x,r.p4y,r.p4z,r.separation4,r.anchorB4X,r.anchorB4Y,r.anchorB4Z,r.normalImpulse4,r.feature4};
      prepareTable[contactId]=prep;r.residentFlags|=2u;}}
  uint coldTransition=b3_manifold_cold_transition(r,inputs[i],p);if(coldTransition!=0u){PrepareInput prep={};prep.contactId=contactId;prep.indexA=transformA.index;prep.indexB=transformB.index;prep.generation=p.currentGeneration;prep.manifold=0ul;
    prep.friction=r.friction;prep.restitution=r.restitution;prep.rollingResistance=r.rollingResistance;prep.tangentVelocityX=r.tangentVelocityX;prep.tangentVelocityY=r.tangentVelocityY;prep.tangentVelocityZ=r.tangentVelocityZ;prep.contactGeneration=inputs[i].contactGeneration;
    prep.points[0]=PreparePoint{r.p1x,r.p1y,r.p1z,r.separation1,r.anchorB1X,r.anchorB1Y,r.anchorB1Z,0.0f,r.feature1};prep.points[1]=PreparePoint{r.p2x,r.p2y,r.p2z,r.separation2,r.anchorB2X,r.anchorB2Y,r.anchorB2Z,0.0f,r.feature2};
    prep.points[2]=PreparePoint{r.p3x,r.p3y,r.p3z,r.separation3,r.anchorB3X,r.anchorB3Y,r.anchorB3Z,0.0f,r.feature3};prep.points[3]=PreparePoint{r.p4x,r.p4y,r.p4z,r.separation4,r.anchorB4X,r.anchorB4Y,r.anchorB4Z,0.0f,r.feature4};prepareTable[contactId]=prep;inputs[i].prepareEligible|=2u;r.residentFlags|=2u;
    if(p.p1==2u)transitions[blocks[i/256u].transitionOffset+transitionLocal]=contactId;else{transitions[2u*contactId]=inputs[i].contactGeneration;transitions[2u*contactId+1u]=r.pointCount;}}
  else if(p.p1!=0u){uint output=blocks[i/256u].exceptionOffset+r.scanOffset;r.inputIndex=i;r.scanOffset=0u;r.contactId=contactId;compact[output]=r;}
  else if(p.p1==0u){uint stable=p.p0!=0u&&(inputs[i].prepareEligible&2u)!=0u&&(r.residentFlags&6u)==2u&&r.touching!=0u&&r.pointCount>0u&&r.pointCount<=4u;if(p.p0==0u||stable==0u){uint output=blocks[i/256u].exceptionOffset+r.scanOffset;r.inputIndex=i;r.scanOffset=0u;r.contactId=contactId;compact[output]=r;}}
  r.inputIndex=contactId;r.contactId=contactId;table[contactId]=r;}
