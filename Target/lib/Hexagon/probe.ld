/* Copyright (c) 2026, Modular Inc. All rights reserved.
 * Licensed under the MAX License: https://modular.com/legal/max-license */
/* QuRT's DSP loader requires separate page-aligned executable/data segments. */
PHDRS {
  code PT_LOAD FILEHDR PHDRS FLAGS(5);
  data PT_LOAD FLAGS(6);
  dynamic PT_DYNAMIC FLAGS(6);
}
SECTIONS {
  . = SIZEOF_HEADERS;
  .hash : { *(.hash) } :code
  .gnu.hash : { *(.gnu.hash) } :code
  .dynsym : { *(.dynsym) } :code
  .dynstr : { *(.dynstr) } :code
  .text : { *(.text*) } :code
  .rodata : { *(.rodata*) } :code
  .rxpad : { BYTE(0); . = ALIGN(4096); } :code
  .data : { *(.data*) *(.got*) } :data
  .dynamic : { *(.dynamic) } :data :dynamic
  .bss : { *(.bss*) *(COMMON) } :data
  .rwpad : { BYTE(0); . = ALIGN(4096); } :data
  /DISCARD/ : { *(.eh_frame*) *(.comment) }
}
