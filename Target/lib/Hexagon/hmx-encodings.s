// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the MAX License: https://modular.com/legal/max-license
.text
{ mxclracc.hf }
{ activation.hf = mxmem(r0, r1)
  weight.hf = mxmem(r2, r3) }
{ bias = mxmem2(r4) }
{ mxmem(r5,r6):after.hf = acc }
