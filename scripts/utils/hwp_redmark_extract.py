#!/usr/bin/env python3
"""HWP(한글 5.0) 초안에서 빨간 글씨/밑줄/형광펜 표시를 추출한다.

교수님 첨삭이 .docx가 아니라 .hwp로 올 때 쓰는 파서. LibreOffice는 이 파일을
열지 못하므로(source file could not be loaded) OLE 스트림을 직접 읽는다.

동작:
  DocInfo의 CharShape(tag 21) 레코드에서 글자색·음영색·밑줄 속성을 읽고,
  BodyText/Section*의 PARA_TEXT(67) + PARA_CHAR_SHAPE(68)로 문자 위치에
  서식을 매핑해 색상이 바뀌는 구간을 뽑는다.

HWP5 CharShape 레이아웃 (오프셋 주의 — 잘못 잡으면 전부 같은 색으로 나온다):
  0..13  faceNameIds[7]     14..20 ratios[7]      21..27 charSpacings[7]
  28..34 relSizes[7]        35..41 charOffsets[7] 42..45 baseSize(INT32)
  46..49 property(UINT32)   50..51 shadowGap      52..55 textColor(COLORREF)
  56..59 shadeColor         60..63 shadowColor
  COLORREF는 0x00BBGGRR 리틀엔디안 → byte0=R, byte1=G, byte2=B
  property: bit0 italic, bit1 bold, bit2~3 밑줄종류

Usage:
  python3 scripts/utils/hwp_redmark_extract.py <파일.hwp>

의존: pip install olefile
"""
import olefile, zlib, struct, sys, re
P=sys.argv[1]
o=olefile.OleFileIO(P)
def raw(n): return zlib.decompress(o.openstream(n).read(), -15)
def records(buf):
    i=0
    while i+4<=len(buf):
        h=struct.unpack('<I',buf[i:i+4])[0]
        tag=h&0x3FF; sz=(h>>20)&0xFFF; i+=4
        if sz==0xFFF: sz=struct.unpack('<I',buf[i:i+4])[0]; i+=4
        yield tag,buf[i:i+sz]; i+=sz
def cref(b): return (b[0],b[1],b[2])
shapes=[]
for tag,d in records(raw('DocInfo')):
    if tag==21:
        if len(d)<60: shapes.append(None); continue
        prop=struct.unpack('<I',d[46:50])[0]
        shapes.append({'text':cref(d[52:56]),'und':(prop>>2)&0x3,'bold':(prop>>1)&1})
EXT={1,2,3,11,12,14,15,16,17,18,21,22,23}; INL={4,5,6,7,8,9,19,20}
def pchars(d):
    r=[];i=0;pos=0
    while i+2<=len(d):
        c=struct.unpack('<H',d[i:i+2])[0]
        if c in EXT: r.append((pos,'[OBJ]')); i+=16; pos+=8
        elif c in INL: r.append((pos,'')); i+=16; pos+=8
        elif c in (10,13): r.append((pos,'\n')); i+=2; pos+=1
        elif c<32: r.append((pos,'')); i+=2; pos+=1
        else: r.append((pos,chr(c))); i+=2; pos+=1
    return r
def sat(cs,p):
    s=None
    for pp,ss in cs:
        if pp<=p: s=ss
        else: break
    return s
paras=[]
for sec in ['BodyText/Section0','BodyText/Section1']:
    buf=raw(sec); t=None; cs=None
    for tag,d in records(buf):
        if tag==66:
            if t is not None: paras.append((sec,t,cs))
            t=None; cs=None
        elif tag==67: t=pchars(d)
        elif tag==68:
            cs=[struct.unpack('<Ii',d[k*8:k*8+8]) for k in range(len(d)//8)]
    if t is not None: paras.append((sec,t,cs))

plains=[''.join(c for _,c in t) for _,t,_ in paras]
RED=(255,0,0)
print("="*74); print("빨간 글씨 4건 — 전후 맥락"); print("="*74)
for target in [231,268,272,274]:
    print(f'\n───── p{target} ─────')
    for j in range(max(0,target-4), min(len(plains),target+4)):
        sec,t,cs=paras[j]
        txt=plains[j].strip()
        if not txt: continue
        isred=False
        if cs:
            for p,c in t:
                sid=sat(cs,p)
                if sid is not None and 0<=sid<len(shapes) and shapes[sid] and shapes[sid]['text']==RED and c.strip():
                    isred=True; break
        mark='🔴' if isred else ('  ' if j!=target else '👉')
        print(f'{mark} [p{j}] {txt[:150]}')
