#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

// undefined8 FUN_00043050(undefined8 param_1,undefined8 *param_2,ulong param_3,long param_4,undefined8 *param_5, int param_6) { uint *puVar1; uint uVar2; bool bVar3; undefined8 uVar4; ulong uVar5; ulong uVar6; long lVar7; undefined8 uVar8; undefined8 uVar9; undefined8 uVar10; if (param_5 == (undefined8 *)0x0) { if (DAT_00526768 != 0) { FUN_0003e580(); } FUN_00022ef0("ASSERT: ","./sne/drivers/td/CSneTDDrvH13.cpp",0x83,"%s\n"); do { /* WARNING: Do nothing block with infinite loop */ } while( true ); } if ((param_2 != (undefined8 *)0x0) && (0x27 < param_3)) { if (param_4 == 0) { if (DAT_00526768 != 0) { FUN_0003e580(); } FUN_00022ef0("ASSERT: ","./sne/drivers/td/CSneTDDrvH13.cpp",0x85,"%s\n"); do { /* WARNING: Do nothing block with infinite loop */ } while( true ); } FUN_00005ecc(param_5,0,0x2c); uVar8 = param_2[2]; uVar4 = param_2[4]; uVar10 = param_2[1]; uVar9 = *param_2; param_5[3] = param_2[3]; param_5[2] = uVar8; param_5[1] = uVar10; *param_5 = uVar9; param_5[4] = uVar4; if ((*(byte *)((long)param_5 + 0x1b) & 1) != 0) { if (param_3 < 0x2c) { if (DAT_00526768 != 0) { FUN_0003e580(); } FUN_00022ef0("ASSERT: ","./sne/drivers/td/CSneTDDrvH13.cpp",0xcf,"%s\n"); do { /* WARNING: Do nothing block with infinite loop */ } while( true ); } *(undefined4 *)(param_5 + 5) = *(undefined4 *)(param_2 + 5); } uVar6 = (ulong)(*(uint *)(param_2 + 3) >> 0x16) & 4 | 0x28; if (param_6 == 0) { for (; uVar6 < param_3; uVar6 = (ulong)((int)uVar6 + (uVar2 >> 0x1a) * 4 + 8)) { puVar1 = (uint *)((long)param_2 + uVar6); FUN_00022ad4("regCount %d\n"); uVar2 = *puVar1; uVar5 = 0; do { *(uint *)(param_4 + (uVar5 + (uVar2 >> 2 & 0xffffff)) * 4) = puVar1[uVar5 + 1]; uVar2 = *puVar1; bVar3 = uVar5 < uVar2 >> 0x1a; uVar5 = uVar5 + 1; } while (bVar3); } } else { FUN_00022ad4("td size %zu while usedSize %d\n"); for (; uVar6 < param_3; uVar6 = (ulong)((int)uVar6 + (uVar2 >> 0x1a) * 4 + 8)) { puVar1 = (uint *)((long)param_2 + uVar6); FUN_00022ad4("regCount %d\n"); lVar7 = 0; uVar2 = *puVar1; uVar5 = 0xffffffffffffffff; do { *(undefined4 *)(param_4 + (uVar5 + (uVar2 >> 2 & 0xffffff)) * 4 + 4) = *(undefined4 *)((long)puVar1 + lVar7 + 4); FUN_00022ad4("reg addr 0x%x with value 0x%x\n"); uVar2 = *puVar1; lVar7 = lVar7 + 4; uVar5 = uVar5 + 1; } while (uVar5 < uVar2 >> 0x1a); FUN_00022ad4("td size %zu while usedSize %d\n"); } } return 1; } if (DAT_00526768 != 0) { FUN_0003e580(); } FUN_00022ef0("ASSERT: ","./sne/drivers/td/CSneTDDrvH13.cpp",0x84,"%s\n"); do { /* WARNING: Do nothing block with infinite loop */ } while( true ); }

void DumpTDRegBlocks(const void *tdData, size_t tdSize);

/**
 * Parse a TD (Task Descriptor) and optionally apply its register writes
 *
 * param_1 -> TD driver context (unused here)
 * param_2 -> Pointer to TD binary
 * param_3 -> TD size in bytes
 * param_4 -> Register base address
 * param_5 -> Output TD header (parsed)
 * param_6 -> Mode:
 *            0 = write registers directly
 *            1 = write registers with +4 offset (alternate mode)
 *
 * Returns 1 on success (never returns on failure)
 */
uint64_t ParseTD(
    uint64_t   ctx,
    uint64_t  *tdData,
    uint64_t   tdSize,
    uint64_t   regBase,
    uint64_t  *outHeader,
    int        altMode
)
{
    /* ---------- Required output ---------- */

    if (outHeader == NULL) {
        ASSERT("outHeader is NULL");
    }

    /* ---------- Validate input ---------- */

    if (tdData == NULL || tdSize <= 0x27) {
        ASSERT("invalid TD data or size");
    }

    if (regBase == 0) {
        ASSERT("regBase is NULL");
    }

    /* ---------- Copy TD header ---------- */

    memset(outHeader, 0, 0x2C);

    outHeader[0] = tdData[0];
    outHeader[1] = tdData[1];
    outHeader[2] = tdData[2];
    outHeader[3] = tdData[3];
    outHeader[4] = tdData[4];

    /* Extended TD header */
    if (*(uint8_t *)((uint8_t *)outHeader + 0x1B) & 0x01) {
        if (tdSize < 0x2C) {
            ASSERT("extended TD but size < 0x2C");
        }
        *(uint32_t *)(outHeader + 5) = *(uint32_t *)(tdData + 5);
    }

    /* ---------- Register list parsing ---------- */

    uint64_t offset =
        ((*(uint32_t *)(tdData + 3) >> 22) & 0x04) | 0x28;

#ifdef TD_DUMP_BLOCKS
    DumpTDRegBlocks(tdData, (size_t)tdSize);
#endif

    if (altMode == 0) {

        /* Normal register write mode */
        while (offset < tdSize) {

            uint32_t *regBlock = (uint32_t *)((uint8_t *)tdData + offset);
            uint32_t header    = regBlock[0];

            uint32_t regCount  = header >> 26;
            uint32_t regBaseId = (header >> 2) & 0x00FFFFFF;

            for (uint32_t i = 0; i < regCount; i++) {
                *(uint32_t *)(regBase + (regBaseId + i) * 4) =
                    regBlock[i + 1];
            }

            offset += regCount * 4 + 8;
        }
    }
    else {

        /* Alternate register write mode */
        while (offset < tdSize) {

            uint32_t *regBlock = (uint32_t *)((uint8_t *)tdData + offset);
            uint32_t header    = regBlock[0];

            uint32_t regCount  = header >> 26;
            uint32_t regBaseId = (header >> 2) & 0x00FFFFFF;

            for (uint32_t i = 0; i < regCount; i++) {
                *(uint32_t *)(regBase + (regBaseId + i) * 4 + 4) =
                    regBlock[i + 1];
            }

            offset += regCount * 4 + 8;
        }
    }

    return 1;
}

/**
 * Dump TD register block metadata for mapping new/unknown groups.
 *
 * Prints: offset, regBaseId, regCount, valueCount, nonZero count.
 */
void DumpTDRegBlocks(const void *tdData, size_t tdSize)
{
    if (tdData == NULL || tdSize <= 0x27) {
        printf("DumpTDRegBlocks: invalid TD data/size\n");
        return;
    }

    const uint8_t *bytes = (const uint8_t *)tdData;
    uint64_t offset = ((*(const uint32_t *)(bytes + 3 * sizeof(uint64_t)) >> 22) & 0x04) | 0x28;
    uint32_t blockIdx = 0;

    printf("---- TD reg blocks ----\n");
    while (offset < tdSize) {
        size_t left = tdSize - (size_t)offset;
        if (left < 4) {
            printf("block %u: trailing %zu bytes at 0x%zx\n", blockIdx, left, (size_t)offset);
            break;
        }

        const uint32_t *regBlock = (const uint32_t *)(bytes + offset);
        uint32_t header = regBlock[0];
        uint32_t regCount = header >> 26;
        uint32_t regBaseId = (header >> 2) & 0x00FFFFFF;
        uint32_t valueCount = regCount + 1; /* per decompile: do-while writes regCount+1 values */

        size_t blockBytes = 8 + (size_t)valueCount * 4;
        size_t valuesAvail = (left >= 8) ? (left - 8) / 4 : 0;
        if (valuesAvail < valueCount) {
            valueCount = (uint32_t)valuesAvail;
        }

        uint32_t nonZero = 0;
        for (uint32_t i = 0; i < valueCount; ++i) {
            if (regBlock[i + 1] != 0) {
                nonZero++;
            }
        }

        printf("block %u: off=0x%zx regBase=0x%06x regCount=%u values=%u nonZero=%u",
               blockIdx, (size_t)offset, regBaseId, regCount, valueCount, nonZero);
        if (left < blockBytes) {
            printf(" (truncated, left=%zu need=%zu)", left, blockBytes);
        }
        printf("\n");

        offset += blockBytes;
        blockIdx++;
    }
}




undefined8
FUN_00041204(undefined8 param_1,undefined8 *param_2,ulong param_3,long param_4,undefined8 *param_5)

{
  uint *puVar1;
  uint uVar2;
  bool bVar3;
  ulong uVar4;
  undefined8 uVar5;
  ulong uVar6;
  undefined8 uVar7;
  undefined8 uVar8;
  undefined8 uVar9;
  
  if (param_5 == (undefined8 *)0x0) {
    if (DAT_00526768 != 0) {
      FUN_0003e580();
    }
    FUN_00022ef0("ASSERT: ","./sne/drivers/tm/CSneTMDrvH13.cpp",0x7e,"%s\n");
    do {
                    /* WARNING: Do nothing block with infinite loop */
    } while( true );
  }
  if ((param_2 != (undefined8 *)0x0) && (0x27 < param_3)) {
    if (param_4 == 0) {
      if (DAT_00526768 != 0) {
        FUN_0003e580();
      }
      FUN_00022ef0("ASSERT: ","./sne/drivers/tm/CSneTMDrvH13.cpp",0x80,"%s\n");
      do {
                    /* WARNING: Do nothing block with infinite loop */
      } while( true );
    }
    FUN_00005ecc(param_5,0,0x2c);
    uVar7 = param_2[2];
    uVar5 = param_2[4];
    uVar9 = param_2[1];
    uVar8 = *param_2;
    param_5[3] = param_2[3];
    param_5[2] = uVar7;
    param_5[1] = uVar9;
    *param_5 = uVar8;
    param_5[4] = uVar5;
    if ((*(byte *)((long)param_5 + 0x1b) & 1) != 0) {
      if (param_3 < 0x2c) {
        if (DAT_00526768 != 0) {
          FUN_0003e580();
        }
        FUN_00022ef0("ASSERT: ","./sne/drivers/tm/CSneTMDrvH13.cpp",0xc9,"%s\n");
        do {
                    /* WARNING: Do nothing block with infinite loop */
        } while( true );
      }
      *(undefined4 *)(param_5 + 5) = *(undefined4 *)(param_2 + 5);
    }
    for (uVar4 = (ulong)(*(uint *)(param_2 + 3) >> 0x16) & 4 | 0x28; uVar4 < param_3;
        uVar4 = (ulong)((int)uVar4 + (uVar2 >> 0x1a) * 4 + 8)) {
      puVar1 = (uint *)((long)param_2 + uVar4);
      uVar2 = *puVar1;
      uVar6 = 0;
      do {
        *(uint *)(param_4 + (uVar6 + (uVar2 >> 2 & 0xffffff)) * 4) = puVar1[uVar6 + 1];
        uVar2 = *puVar1;
        bVar3 = uVar6 < uVar2 >> 0x1a;
        uVar6 = uVar6 + 1;
      } while (bVar3);
    }
    return 1;
  }
  if (DAT_00526768 != 0) {
    FUN_0003e580();
  }
  FUN_00022ef0("ASSERT: ","./sne/drivers/tm/CSneTMDrvH13.cpp",0x7f,"%s\n");
  do {
                    /* WARNING: Do nothing block with infinite loop */
  } while( true );
}




undefined8
FUN_00043050(undefined8 param_1,undefined8 *param_2,ulong param_3,long param_4,undefined8 *param_5,
            int param_6)

{
  uint *puVar1;
  uint uVar2;
  bool bVar3;
  undefined8 uVar4;
  ulong uVar5;
  ulong uVar6;
  long lVar7;
  undefined8 uVar8;
  undefined8 uVar9;
  undefined8 uVar10;
  
  if (param_5 == (undefined8 *)0x0) {
    if (DAT_00526768 != 0) {
      FUN_0003e580();
    }
    FUN_00022ef0("ASSERT: ","./sne/drivers/td/CSneTDDrvH13.cpp",0x83,"%s\n");
    do {
                    /* WARNING: Do nothing block with infinite loop */
    } while( true );
  }
  if ((param_2 != (undefined8 *)0x0) && (0x27 < param_3)) {
    if (param_4 == 0) {
      if (DAT_00526768 != 0) {
        FUN_0003e580();
      }
      FUN_00022ef0("ASSERT: ","./sne/drivers/td/CSneTDDrvH13.cpp",0x85,"%s\n");
      do {
                    /* WARNING: Do nothing block with infinite loop */
      } while( true );
    }
    FUN_00005ecc(param_5,0,0x2c);
    uVar8 = param_2[2];
    uVar4 = param_2[4];
    uVar10 = param_2[1];
    uVar9 = *param_2;
    param_5[3] = param_2[3];
    param_5[2] = uVar8;
    param_5[1] = uVar10;
    *param_5 = uVar9;
    param_5[4] = uVar4;
    if ((*(byte *)((long)param_5 + 0x1b) & 1) != 0) {
      if (param_3 < 0x2c) {
        if (DAT_00526768 != 0) {
          FUN_0003e580();
        }
        FUN_00022ef0("ASSERT: ","./sne/drivers/td/CSneTDDrvH13.cpp",0xcf,"%s\n");
        do {
                    /* WARNING: Do nothing block with infinite loop */
        } while( true );
      }
      *(undefined4 *)(param_5 + 5) = *(undefined4 *)(param_2 + 5);
    }
    uVar6 = (ulong)(*(uint *)(param_2 + 3) >> 0x16) & 4 | 0x28;
    if (param_6 == 0) {
      for (; uVar6 < param_3; uVar6 = (ulong)((int)uVar6 + (uVar2 >> 0x1a) * 4 + 8)) {
        puVar1 = (uint *)((long)param_2 + uVar6);
        FUN_00022ad4("regCount %d\n");
        uVar2 = *puVar1;
        uVar5 = 0;
        do {
          *(uint *)(param_4 + (uVar5 + (uVar2 >> 2 & 0xffffff)) * 4) = puVar1[uVar5 + 1];
          uVar2 = *puVar1;
          bVar3 = uVar5 < uVar2 >> 0x1a;
          uVar5 = uVar5 + 1;
        } while (bVar3);
      }
    }
    else {
      FUN_00022ad4("td size %zu while usedSize %d\n");
      for (; uVar6 < param_3; uVar6 = (ulong)((int)uVar6 + (uVar2 >> 0x1a) * 4 + 8)) {
        puVar1 = (uint *)((long)param_2 + uVar6);
        FUN_00022ad4("regCount %d\n");
        lVar7 = 0;
        uVar2 = *puVar1;
        uVar5 = 0xffffffffffffffff;
        do {
          *(undefined4 *)(param_4 + (uVar5 + (uVar2 >> 2 & 0xffffff)) * 4 + 4) =
               *(undefined4 *)((long)puVar1 + lVar7 + 4);
          FUN_00022ad4("reg addr 0x%x with value 0x%x\n");
          uVar2 = *puVar1;
          lVar7 = lVar7 + 4;
          uVar5 = uVar5 + 1;
        } while (uVar5 < uVar2 >> 0x1a);
        FUN_00022ad4("td size %zu while usedSize %d\n");
      }
    }
    return 1;
  }
  if (DAT_00526768 != 0) {
    FUN_0003e580();
  }
  FUN_00022ef0("ASSERT: ","./sne/drivers/td/CSneTDDrvH13.cpp",0x84,"%s\n");
  do {
                    /* WARNING: Do nothing block with infinite loop */
  } while( true );
}


undefined8 FUN_00043880(undefined8 param_1,uint *param_2,ulong param_3,int param_4)

{
  uint uVar1;
  
  if ((param_2 == (uint *)0x0) || (param_3 < 0x28)) {
    if (DAT_00526768 != 0) {
      FUN_0003e580();
    }
    FUN_00022ef0("ASSERT: ","./sne/drivers/td/CSneTDDrvH13.cpp",0x169,"%s\n");
    do {
                    /* WARNING: Do nothing block with infinite loop */
    } while( true );
  }
  if (0xfe < param_4 - 1U) {
    if (DAT_00526768 != 0) {
      FUN_0003e580();
    }
    FUN_00022ef0("ASSERT: ","./sne/drivers/td/CSneTDDrvH13.cpp",0x171,"%s\n");
    do {
                    /* WARNING: Do nothing block with infinite loop */
    } while( true );
  }
  uVar1 = *param_2;
  *param_2 = uVar1 & 0xff00ffff | param_4 << 0x10;
  if ((uVar1 >> 0x18 & 1) != 0) {
    if (DAT_00526768 != 0) {
      FUN_0003e580();
    }
    FUN_00022d8c("MSG: ","./sne/drivers/td/CSneTDDrvH13.cpp",0x173,"invalid LNID %d for ptd %p");
    if ((*param_2 & 0x1000000) != 0) {
      FUN_00022ef0("ASSERT: ","./sne/drivers/td/CSneTDDrvH13.cpp",0x173,"%s\n");
      do {
                    /* WARNING: Do nothing block with infinite loop */
      } while( true );
    }
  }
  FUN_00094e00(param_2,param_3);
  return 1;
}


uint FUN_00043a04(undefined8 param_1,uint *param_2,ulong param_3)

{
  uint uVar1;
  
  if ((param_2 == (uint *)0x0) || (param_3 < 0x28)) {
    if (DAT_00526768 != 0) {
      FUN_0003e580();
    }
    FUN_00022ef0("ASSERT: ","./sne/drivers/td/CSneTDDrvH13.cpp",0x17b,"%s\n");
    do {
                    /* WARNING: Do nothing block with infinite loop */
    } while( true );
  }
  uVar1 = *param_2;
  if ((uVar1 & 0xff0000) == 0) {
    if (DAT_00526768 != 0) {
      FUN_0003e580();
      uVar1 = *param_2;
    }
    if ((uVar1 & 0xff0000) == 0) {
      FUN_00022ef0("ASSERT: ","./sne/drivers/td/CSneTDDrvH13.cpp",0x185,"%s\n");
      do {
                    /* WARNING: Do nothing block with infinite loop */
      } while( true );
    }
  }
  return uVar1 >> 0x10 & 0xff;
}


undefined8 FUN_0006d018(uint *param_1,ulong param_2,long param_3,ulong param_4)

{
  ushort *puVar1;
  uint uVar2;
  uint uVar3;
  undefined8 uVar4;
  char *pcVar5;
  ulong uVar6;
  uint *puVar7;
  
  if (param_1 == (uint *)0x0) {
    if (DAT_00526768 != 0) {
      FUN_0003e580();
    }
    FUN_00022ef0("ASSERT: ","./sne/aneEngine/program/CAneProgramCheckerH13.cpp",0xef,"%s\n");
    do {
                    /* WARNING: Do nothing block with infinite loop */
    } while( true );
  }
  if (param_3 == 0) {
    if (DAT_00526768 != 0) {
      FUN_0003e580();
    }
    FUN_00022ef0("ASSERT: ","./sne/aneEngine/program/CAneProgramCheckerH13.cpp",0xf0,"%s\n");
    do {
                    /* WARNING: Do nothing block with infinite loop */
    } while( true );
  }
  if (param_2 < ((ulong)*param_1 * 0x30 | 4)) {
    pcVar5 = "\x1b[31m[VERIFICATION]\x1b[39m TdProp section (%lu) is smaller than actual (%lu)!";
    uVar4 = 0xf6;
  }
  else {
    if (*param_1 == 0) {
      return 1;
    }
    if (param_4 < param_1[3] + param_1[2]) {
LAB_0006d134:
      pcVar5 = 
      "\x1b[31m[VERIFICATION]\x1b[39m TD[%d] exceeds limit (offset, len) (0x%x, %d) section size %lu !"
      ;
      uVar4 = 0x107;
    }
    else {
      puVar1 = (ushort *)(param_3 + (ulong)param_1[2]);
      uVar3 = 0x28;
      if ((*(uint *)(puVar1 + 0xc) & 0x1000000) != 0) {
        uVar3 = 0x2c;
      }
      if (param_1[3] < uVar3) {
        FUN_00022d8c("ERR: ","./sne/aneEngine/program/CAneProgramCheckerH13.cpp",0x111,
                     "\x1b[31m[VERIFICATION]\x1b[39m TD[%d] len %d is smaller than ane_TD_HEADER_t ( TDE %d)!"
                    );
      }
      if ((uint)*puVar1 == param_1[1]) {
        if (*param_1 < 2) {
          return 1;
        }
        puVar7 = param_1 + 0xf;
        uVar6 = 1;
        while( true ) {
          uVar3 = puVar7[-1];
          if (uVar3 < puVar7[-0xc] + puVar7[-0xd]) break;
          if (param_4 < *puVar7 + uVar3) goto LAB_0006d134;
          uVar2 = 0x28;
          if ((*(uint *)((ushort *)(param_3 + (ulong)uVar3) + 0xc) & 0x1000000) != 0) {
            uVar2 = 0x2c;
          }
          if (*puVar7 < uVar2) {
            FUN_00022d8c("ERR: ","./sne/aneEngine/program/CAneProgramCheckerH13.cpp",0x111,
                         "\x1b[31m[VERIFICATION]\x1b[39m TD[%d] len %d is smaller than ane_TD_HEADER _t (TDE %d)!"
                        );
          }
          if ((uint)*(ushort *)(param_3 + (ulong)uVar3) != puVar7[-2]) goto LAB_0006d2b4;
          uVar6 = uVar6 + 1;
          puVar7 = puVar7 + 0xc;
          if (*param_1 <= uVar6) {
            return 1;
          }
        }
        pcVar5 = 
        "\x1b[31m[VERIFICATION]\x1b[39m TD[%d] offset 0x%x is overlapped with the previous offset 0x %x len %d!"
        ;
        uVar4 = 0xff;
      }
      else {
LAB_0006d2b4:
        pcVar5 = "\x1b[31m[VERIFICATION]\x1b[39m TD[%d] (headerTID:TdPropTID:index)=(%d:%d)";
        uVar4 = 0x116;
      }
    }
  }
  FUN_00022d8c("ERR: ","./sne/aneEngine/program/CAneProgramCheckerH13.cpp",uVar4,pcVar5);
  return 0;
}

