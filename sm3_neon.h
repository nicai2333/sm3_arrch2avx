#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "sm3.h"
#include "config.h"

// 假设有某些初始化和数据预处理部分

uint32_t ll_bswap4(uint32_t a) {
    uint32x2_t tmp = vdup_n_u32(a); // 将 a 复制到 NEON 寄存器的两个32位元素中
    uint8x8_t reversed = vrev32_u8(vreinterpret_u8_u32(tmp)); // 使用 vrev32_u8 对每个32位块的字节进行反转
    return vget_lane_u32(vreinterpret_u32_u8(reversed), 0); // 提取反转后的结果
}

// 字节翻转函数，用于 64 位无符号整数
uint64_t ll_bswap8(uint64_t a) {
    uint64x1_t tmp = vdup_n_u64(a); // 将 a 复制到 NEON 寄存器
    uint8x8_t reversed = vrev64_u8(vreinterpret_u8_u64(tmp)); // 使用 vrev64_u8 对64位的字节进行反转
    return vget_lane_u64(vreinterpret_u64_u8(reversed), 0); // 提取反转后的结果
}

static const uint32_t Tj[] = {
    0x79CC4519, 0xF3988A32, 0xE7311465, 0xCE6228CB,
    0x9CC45197, 0x3988A32F, 0x7311465E, 0xE6228CBC,
    0xCC451979, 0x988A32F3, 0x311465E7, 0x6228CBCE,
    0xC451979C, 0x88A32F39, 0x11465E73, 0x228CBCE6,
    0x9D8A7A87, 0x3B14F50F, 0x7629EA1E, 0xEC53D43C,
    0xD8A7A879, 0xB14F50F3, 0x629EA1E7, 0xC53D43CE,
    0x8A7A879D, 0x14F50F3B, 0x29EA1E76, 0x53D43CEC,
    0xA7A879D8, 0x4F50F3B1, 0x9EA1E762, 0x3D43CEC5,
    0x7A879D8A, 0xF50F3B14, 0xEA1E7629, 0xD43CEC53,
    0xA879D8A7, 0x50F3B14F, 0xA1E7629E, 0x43CEC53D,
    0x879D8A7A, 0x0F3B14F5, 0x1E7629EA, 0x3CEC53D4,
    0x79D8A7A8, 0xF3B14F50, 0xE7629EA1, 0xCEC53D43,
    0x9D8A7A87, 0x3B14F50F, 0x7629EA1E, 0xEC53D43C,
    0xD8A7A879, 0xB14F50F3, 0x629EA1E7, 0xC53D43CE,
    0x8A7A879D, 0x14F50F3B, 0x29EA1E76, 0x53D43CEC,
    0xA7A879D8, 0x4F50F3B1, 0x9EA1E762, 0x3D43CEC5
};
//int j=0;

//turn right
uint32x4_t ROR32(uint32x4_t input, int shift) {
    const int n = shift % 32;
    const int m = 32 - n;
    
    uint8x16_t temp = vreinterpretq_u8_u32(input);
    //uint8x16_t rotated = vextq_u8(temp, temp, m / 8);
    
    uint8x16_t shifted_left = vshlq_n_u8(temp, n );
    uint8x16_t shifted_right = vshrq_n_u8(temp, m );
    
    uint8x16_t result = vorrq_u8(shifted_left, shifted_right);
    
    return vreinterpretq_u32_u8(result);
}

uint32_t rotate_right(uint32_t n, int shift) {
    return (n >> shift) | (n << (32 - shift));
}

void circular_shift(uint32x4_t M0, uint32x4_t M1, uint32x4_t M2, uint32x4_t M3) {
    uint32x4_t first_element = M0;  // 保存第一个元素
        M0 = M1;  // 后面的元素向前移动一个位置
        M1 = M2;
        M2 = M3;
        M3 = first_element; // 将保存的第一个元素放到数组末尾
}

void FIRST_16_ROUNDS_AND_SCHED_1(uint32x4_t X0, uint32x4_t X1, uint32x4_t X2, uint32x4_t X3, uint32_t A, uint32_t B, uint32_t C, uint32_t D,uint32_t E, uint32_t F, uint32_t G, uint32_t H,\
uint32x4_t XFER, uint32x4_t XTMP0, uint32x4_t XTMP1,uint32x4_t XTMP2, uint32x4_t XTMP3, uint32x4_t XTMP4, uint32x4_t XTMP5, int j)
{
    XFER = veorq_u32(X0, X1);                // WW
    uint32_t t0 = rotate_right(A, 20);                         // A <<< 12 (Adjusted to 20 based on your input)
    XTMP0 = vextq_u32(X0, X1, 3);            // (W[-13],W[-12],W[-11],XXX)
    uint32_t t1 = A ^ B;                                // A ^ B
    uint32_t t3 = Tj[j];   
    j +=1;          
    uint32_t t2 = t0 + E;                               // (A <<< 12) + E
    uint32_t W = vgetq_lane_u32(X0, 0);                 // W[-16]
    uint32_t t5 = E ^ F;                                // E ^ F
    t1 = t1 ^ C;                                        // FF(A, B, C)
    uint32_t t4 = t2 + t3;                              // (A <<< 12) + E + (Tj <<< j)
    t5 = t5 ^ G;                                        // GG(E, F, G)
    H = H + W;                                             // H + Wj
    t4 = rotate_right(t4, 25);                                 // SS1
    XTMP0 = ROR32(XTMP0,25);                    // (W[-13],W[-12],W[-11],XXX) <<< 7
    D = D + t1;                                            // FF(A, B, C) + D
    B = rotate_right(B, 23);                                      // B <<< 9
    t1 = t4 + t5;                                // GG(E, F, G) + SS1
    XTMP2 = vextq_u32(X2, X3, 2);                       // (W[-6],W[-5],W[-4],XXX)
    W = vgetq_lane_u32(XFER, 0);                        // WW[-16]
    t2 = t0 ^ t4;                                       // SS2
    H = H + t1;                                            // TT2 = GG(E, F, G) + H + SS1 + Wj
    F = rotate_right(F, 13);                                   // F <<< 19
    XTMP0 = veorq_u32(XTMP0, XTMP2);                    // (W[-6],W[-5],W[-4],XXX)^((W[-13],W[-12],W[-11],XXX) <<< 17)
    t3 = rotate_right(H, 23);
    D = D + t2;                                            // FF(A, B, C) + D + SS2
    H = H ^ rotate_right(H, 15);                                 // P0(TT2)
    XTMP1 = vextq_u32(X3, X2, 1);                       // (W[-3],W[-2],W[-1],XXX)
    D = D + W;                                             // TT1 = FF(A, B, C) + D + SS2 + W'j
    H = H ^ t3;                                           // Final P0(TT2)
}

void FIRST_16_ROUNDS_AND_SCHED_2(uint32x4_t X0, uint32x4_t X1, uint32x4_t X2, uint32x4_t X3, uint32_t A, uint32_t B, uint32_t C, uint32_t D,uint32_t E, uint32_t F, uint32_t G, uint32_t H,\
uint32x4_t XFER, uint32x4_t XTMP0, uint32x4_t XTMP1,uint32x4_t XTMP2, uint32x4_t XTMP3, uint32x4_t XTMP4, uint32x4_t XTMP5, int j)
{
    uint32_t t0 = rotate_right(A, 20);                                   // A <<< 12 (corrected to ROR 20)
    uint32_t t1 = A ^ B;                                          // A ^ B
    uint32_t t3 = Tj[j];                       // Tj <<< j, assume loaded from a memory address
    j+=1;
    uint32_t t2 = t0 + E;                                        // (A <<< 12) + E
    uint32_t W = vgetq_lane_u32(X0, 1);                           // W[-15]
    uint32_t t5 = E ^ F;                                          // E ^ F
    t1 = t1 ^ C;                                                    // FF(A, B, C)
    uint32_t t4 = t2 + t3;                                        // (A <<< 12) + E + (Tj <<< j)
    XTMP1 = ROR32(XTMP1, 17);                              // (W[-3],W[-2],W[-1],XXX) <<< 15
    t5 = t5 ^ G;                                                      // GG(E, F, G)
    H = H + W;                                                       // H + Wj
    t4 = rotate_right(t4, 25);                                           // SS1
    XTMP2 = vextq_u32(X1, X2, 3);                                 // W[-9],W[-8],W[-7],W[-6]
    D = D + t1;                                                      // FF(A, B, C) + D
    B = rotate_right(B, 23);                                             // B <<< 9
    t1 = t4 + t5;                                                  // GG(E, F, G) + SS1
    XTMP2 = veorq_u32(XTMP2, X0);                                 // (W[-9],W[-8],W[-7],W[-6]) ^ (W[-16],W[-15],W[-14],W[-13])
    W = vgetq_lane_u32(X0, 1);                                    // WW[-15], assuming correction needed
    t2 = t0 ^ t4;                                                 // SS2
    H = H + t1;                                                      // TT2 = GG(E, F, G) + H + SS1 + Wj
    F = rotate_right(F, 13);                                             // F <<< 19
    XTMP1 = veorq_u32(XTMP1, XTMP2);                              // Combined XOR operation as shown
    t3 = rotate_right(H, 23);
    D = D + t2;                                                      // FF(A, B, C) + D + SS2
    H = H ^ rotate_right(H, 15);                                           // P0(TT2)
    XTMP3 = ROR32(XTMP1, 17);                    // P1(X), X << 15
    D = D + W;                                                       // TT1 = FF(A, B, C) + D + SS2 + W'j
    H = H ^ t3;                                                     // P0(TT2)
}

void FIRST_16_ROUNDS_AND_SCHED_3(uint32x4_t X0, uint32x4_t X1, uint32x4_t X2, uint32x4_t X3, uint32_t A, uint32_t B, uint32_t C, uint32_t D,uint32_t E, uint32_t F, uint32_t G, uint32_t H,\
uint32x4_t XFER, uint32x4_t XTMP0, uint32x4_t XTMP1,uint32x4_t XTMP2, uint32x4_t XTMP3, uint32x4_t XTMP4, uint32x4_t XTMP5, int j)
{
    uint32_t t0 = rotate_right(A, 20);                                 // A <<< 12 (corrected to ROR 20)
    uint32_t t1 = A ^ B;                                        // A ^ B
    uint32_t t3 = Tj[j];                     // Tj <<< j, assume loaded from a memory address
    j+=1;
    uint32_t t2 = t0 + E;                                      // (A <<< 12) + E
    uint32_t W = vgetq_lane_u32(X0, 2);                         // W[-14]
    uint32_t t5 = E ^ F;                                        // E ^ F
    t1 = t1 ^ C;                                                   // FF(A, B, C)
    uint32_t t4 = t2 + t3;                                      // (A <<< 12) + E + (Tj <<< j)
    t5 = t5 ^ G;                                                    // GG(E, F, G)
    H = H + W;                                                     // H + Wj
    t4 = rotate_right(t4, 25);                                         // SS1
    D = D + t1;                                                    // FF(A, B, C) + D
    B = rotate_right(B, 23);                                           // B <<< 9
    t1 = t4 + t5;                                               // GG(E, F, G) + SS1
    XTMP5 = ROR32(XTMP1, 9);                            // P1(X), X << 23 ^ X >> 9
    W = vgetq_lane_u32(X0, 2);                                  // WW[-14], assuming correction needed
    t2 = t0 ^ t4;                                               // SS2
    H = H + t1;                                                    // TT2 = GG(E, F, G) + H + SS1 + Wj
    F = rotate_right(F, 13);                                           // F <<< 19
    XTMP1 = veorq_u32(XTMP1, XTMP3);                            // P1(X), X ^ (X <<< 15)
    t3 = rotate_right(H, 23);
    D = D + t2;                                                    // FF(A, B, C) + D + SS2
    H = H ^ rotate_right(H, 15);                                         // P0(TT2)
    XTMP1 = veorq_u32(XTMP1, XTMP5);                            // P1(X), X ^ (X <<< 15) ^ (X <<< 23)
    D = D + W;                                                     // TT1 = FF(A, B, C) + D + SS2 + W'j
    H = H ^ t3;                                                   // P0(TT2)
}

void FIRST_16_ROUNDS_AND_SCHED_4(uint32x4_t X0, uint32x4_t X1, uint32x4_t X2, uint32x4_t X3, uint32_t A, uint32_t B, uint32_t C, uint32_t D,uint32_t E, uint32_t F, uint32_t G, uint32_t H,\
uint32x4_t XFER, uint32x4_t XTMP0, uint32x4_t XTMP1,uint32x4_t XTMP2, uint32x4_t XTMP3, uint32x4_t XTMP4, uint32x4_t XTMP5, int j)
{

    uint32_t W = vgetq_lane_u32(X0, 3); // W[-13]
    uint32_t t0 = rotate_right(A, 20);         // A <<< 12
    X0 = veorq_u32(XTMP1, XTMP0);       // W[0], W[1], W[2], XXX
    uint32_t t1 = A ^ B;                // A ^ B
    uint32_t t3 = Tj[j]; // Tj <<< j, assume loaded from a memory address
    j+=1;
    uint32_t t2 = t0 + E;              // (A <<< 12) + E
    uint32_t T0 = vgetq_lane_u32(X0, 0); // W[0]
    uint32_t t5 = E ^ F;                // E ^ F
    t1 = t1 ^ C;                           // FF(A, B, C)
    uint32_t T1 = vgetq_lane_u32(XTMP2, 3); // W[-13] ^ W[-6]
    uint32_t t4 = t2 + t3;              // (A <<< 12) + E + (Tj <<< j)
    t5 = t5 ^ G;                            // GG(E, F, G)
    uint32_t T2 = vgetq_lane_u32(XTMP0, 3); // (W[-10] <<< 7) ^ W[-3]
    T0 = rotate_right(T0,17);
    T1 = T1 ^ T0;            // Z = W[-13] ^ W[-6] ^ (W[0] <<< 15)
    H = H + W;                             // H + Wj
    t4 = rotate_right(t4, 25);                 // SS1
    D = D + t1;                            // FF(A, B, C) + D
    uint32_t T3 = rotate_right(T1, 17);        // Z <<< 15
    B = rotate_right(B, 23);                   // B <<< 9
    t1 = t4 + t5;                       // GG(E, F, G) + SS1
    T1 = T1 ^ rotate_right(T1, 9);                 // Z ^ (Z <<< 23)
    W = vgetq_lane_u32(XFER, 3);        // WW[-13]
    t2 = t0 ^ t4;                       // SS2
    H = H + t1;                            // TT2 = GG(E, F, G) + H + SS1 + Wj
    F = rotate_right(F, 13);                   // F <<< 19
    T1 = T1 ^ T3;                           // Z ^ (Z <<< 15) ^ (Z <<< 23)
    t3 = rotate_right(H, 23);
    D = D + t2;                            // FF(A, B, C) + D + SS2
    H = H ^ rotate_right(H, 15);                 // P0(TT2)
    T2 = T1 ^ T2;                           // W[3]
    D = D + W;                             // TT1 = FF(A, B, C) + D + SS2 + W'j
    H = H ^ t3;                           // P0(TT2)
    X0 = vsetq_lane_u32(T2, X0, 3);     // W[0], W[1], W[2], W[3]
}

void SECOND_36_ROUNDS_AND_SCHED_1(uint32x4_t X0, uint32x4_t X1, uint32x4_t X2, uint32x4_t X3, uint32_t A, uint32_t B, uint32_t C, uint32_t D,uint32_t E, uint32_t F, uint32_t G, uint32_t H,\
uint32x4_t XFER, uint32x4_t XTMP0, uint32x4_t XTMP1,uint32x4_t XTMP2, uint32x4_t XTMP3, uint32x4_t XTMP4, uint32x4_t XTMP5, int j)
{
    uint32_t t0, t1, t2, t3, t4, t5, T0, T1, W;

    // Perform operations
    XFER = veorq_u32(X0, X1);                                      // WW
    t0 = rotate_right(A, 20);                                             // A <<< 12
    XTMP0 = vextq_u32(X0, X1, 3);                                  // (W[-13],W[-12],W[-11],XXX)
    t1 = B | C;                                                    // B | C
    t3 = Tj[j];                                  // Tj <<< j, assume loaded from a memory address
    j+=1;
    t2 = t0 + E;                                                  // (A <<< 12) + E
    T0 = B & C;                                                    // B & C
    //XTMP1 = vshlq_n_u32(XTMP0, 7);                                 // ((W[-13],W[-12],W[-11],XXX) << 7)
    T1 = A & t1;                                                   // A & (B | C)
    W = vgetq_lane_u32(X0, 0);                                     // W[-16]
    t5 = F ^ G;                                                   // F ^ G
    t1 = T0 | T1;                                                 // FF(A, B, C)
    t4 = t2 + t3;                                                 // (A <<< 12) + E + (Tj <<< j)
    //XTMP2 = vshrq_n_u32(XTMP0, 25);                                // (W[-13],W[-12],W[-11],XXX) >> 25
    t5 = t5 & E;                                                      // (F ^ G) & E
    H = H + W;                                                        // H + Wj
    t4 = rotate_right(t4, 25);                                            // SS1
    XTMP0 = ROR32(XTMP0, 25);                               // (W[-13],W[-12],W[-11],XXX] <<< 17
    t5 = t5 ^ G;                                                       // GG(E, F, G)
    D = D + t1;                                                       // FF(A, B, C) + D
    B = rotate_right(B, 23);                                              // B <<< 9
    XTMP2 = vextq_u32(X2, X3, 2);                                  // (W[-6],W[-5],W[-4],XXX)
    t1 = t4 + t5;                                                  // GG(E, F, G) + SS1
    W = vgetq_lane_u32(XFER, 0);                                   // WW[-16]
    t2 = t0 ^ t4;                                                  // SS2
    H = H + t1;                                                       // TT2 = GG(E, F, G) + H + SS1 + Wj
    F = rotate_right(F, 13);                                              // F <<< 19
    XTMP0 = veorq_u32(XTMP0, XTMP2);                               // (W[-6],W[-5],W[-4],XXX)^((W[-13],W[-12],W[-11],XXX) <<< 17)
    t3 = rotate_right(H, 23);
    D = D + t2;                                                       // FF(A, B, C) + D + SS2
    H = H ^ rotate_right(H, 15);
    XTMP1 = vextq_u32(X3, X2, 1);                                  // (W[-3],W[-2],W[-1],XXX)
    D = D + W;                                                        // TT1 = FF(A, B, C) + D + SS2 + W'j
    H = H ^ t3;                                                      // P0(TT2)
}

void SECOND_36_ROUNDS_AND_SCHED_2(uint32x4_t X0, uint32x4_t X1, uint32x4_t X2, uint32x4_t X3, uint32_t A, uint32_t B, uint32_t C, uint32_t D,uint32_t E, uint32_t F, uint32_t G, uint32_t H,\
uint32x4_t XFER, uint32x4_t XTMP0, uint32x4_t XTMP1,uint32x4_t XTMP2, uint32x4_t XTMP3, uint32x4_t XTMP4, uint32x4_t XTMP5, int j)
{
    uint32_t t0, t1, t2, t3, t4, t5, T0, T1, W;

    // Perform operations
    t0 = rotate_right(A, 20);                                             // A <<< 12
    //XTMP2 = vshlq_n_u32(XTMP1, 15);                                // (W[-3],W[-2],W[-1],XXX) << 15
    t1 = B | C;                                                    // B | C
    t3 = Tj[j];                                  // Tj <<< j, assume loaded from a memory address
    j+=1;
    t2 = t0 + E;                                                  // (A <<< 12) + E
    T0 = B & C;                                                    // B & C
    //XTMP1 = vshrq_n_u32(XTMP1, 17);                                // (W[-3],W[-2],W[-1],XXX) >> 17
    T1 = A & t1;                                                   // A & (B | C)
    W = vgetq_lane_u32(X0, 1);                                     // W[-15]
    t5 = F ^ G;                                                   // F ^ G
    t1 = T0 | T1;                                                 // FF(A, B, C)
    t4 = t2 + t3;                                                 // (A <<< 12) + E + (Tj <<< j)
    XTMP1 = ROR32(XTMP1, 17);                               // (W[-3],W[-2],W[-1],XXX) <<< 15
    t5 = t5 & E;                                                      // (F ^ G) & E
    H = H + W;                                                        // H + Wj
    t4 = rotate_right(t4, 25);                                            // SS1
    XTMP2 = vextq_u32(X1, X2, 3);                                  // W[-9],W[-8],W[-7],W[-6]
    t5 = t5 ^ G;                                                       // GG(E, F, G)
    D = D + t1;                                                       // FF(A, B, C) + D
    B = rotate_right(B, 23);                                              // B <<< 9
    XTMP2 = veorq_u32(XTMP2, X0);                                  // (W[-9],W[-8],W[-7],W[-6]) ^ (W[-16],W[-15],W[-14],W[-13])
    t1 = t4 + t5;                                                  // GG(E, F, G) + SS1
    W = vgetq_lane_u32(XFER, 1);                                   // WW[-15]
    t2 = t0 ^ t4;                                                  // SS2
    H = H + t1;                                                       // TT2 = GG(E, F, G) + H + SS1 + Wj
    F = rotate_right(F, 13);                                              // F <<< 19
    XTMP1 = veorq_u32(XTMP1, XTMP2);                               // Complex operation combining previous values
    t3 = rotate_right(H, 23);
    D = D + t2;                                                       // FF(A, B, C) + D + SS2
    H = H ^ rotate_right(H, 15);
    //XTMP3 = vshlq_n_u32(XTMP1, 15);                                // P1(X), X << 15
    D = D + W;                                                        // TT1 = FF(A, B, C) + D + SS2 + W'j
    H = H ^ t3;                                                      // P0(TT2)
}

void SECOND_36_ROUNDS_AND_SCHED_3(uint32x4_t X0, uint32x4_t X1, uint32x4_t X2, uint32x4_t X3, uint32_t A, uint32_t B, uint32_t C, uint32_t D,uint32_t E, uint32_t F, uint32_t G, uint32_t H,\
uint32x4_t XFER, uint32x4_t XTMP0, uint32x4_t XTMP1,uint32x4_t XTMP2, uint32x4_t XTMP3, uint32x4_t XTMP4, uint32x4_t XTMP5, int j)
{
    uint32_t t0, t1, t2, t3, t4, t5, T0, T1, W;

    // Perform operations
    t0 = rotate_right(A, 20);                                             // A <<< 12
    //XTMP4 = vshrq_n_u32(XTMP1, 17);                                // P1(X), X >> 17
    t1 = B | C;                                                    // B | C
    t3 = Tj[j];                                  // Tj <<< j, assume loaded from a memory address
    j+=1;
    t2 = t0 + E;                                                  // (A <<< 12) + E
    T0 = B & C;                                                    // B & C
    XTMP3 = ROR32(XTMP1, 17);                               // P1(X), X <<< 15
    T1 = A & t1;                                                   // A & (B | C)
    W = vgetq_lane_u32(X0, 2);                                     // W[-14]
    t5 = F ^ G;                                                   // F ^ G
    t1 = T0 | T1;                                                 // FF(A, B, C)
    t4 = t2 + t3;                                                 // (A <<< 12) + E + (Tj <<< j)
    //XTMP4 = vshlq_n_u32(XTMP1, 23);                                // P1(X), X << 23
    t5 = t5 & E;                                                      // (F ^ G) & E
    H = H + W;                                                        // H + Wj
    t4 = rotate_right(t4, 25);                                            // SS1
    //XTMP5 = vshrq_n_u32(XTMP1, 9);                                 // P1(X), X >> 9
    t5 = t5 ^ G;                                                       // GG(E, F, G)
    D = D + t1;                                                       // FF(A, B, C) + D
    B = rotate_right(B, 23);                                              // B <<< 9
    XTMP5 = ROR32(XTMP1, 9);                               // P1(X), X << 23
    t1 = t4 + t5;                                                  // GG(E, F, G) + SS1
    W = vgetq_lane_u32(XFER, 2);                                   // WW[-14]
    t2 = t0 ^ t4;                                                  // SS2
    H = H + t1;                                                       // TT2 = GG(E, F, G) + H + SS1 + Wj
    F = rotate_right(F, 13);                                              // F <<< 19
    XTMP1 = veorq_u32(XTMP1, XTMP3);                               // P1(X), X ^ (X <<< 15)
    t3 = rotate_right(H, 23);
    D = D + t2;                                                       // FF(A, B, C) + D + SS2
    H = H ^ rotate_right(H, 15);
    XTMP1 = veorq_u32(XTMP1, XTMP5);                               // P1(X), X ^ (X <<< 15) ^ (X <<< 23)
    D = D + W;                                                        // TT1 = FF(A, B, C) + D + SS2 + W'j
    H = H ^ t3;                                                      // P0(TT2)
}

void SECOND_36_ROUNDS_AND_SCHED_4(uint32x4_t X0, uint32x4_t X1, uint32x4_t X2, uint32x4_t X3, uint32_t A, uint32_t B, uint32_t C, uint32_t D,uint32_t E, uint32_t F, uint32_t G, uint32_t H,\
uint32x4_t XFER, uint32x4_t XTMP0, uint32x4_t XTMP1,uint32x4_t XTMP2, uint32x4_t XTMP3, uint32x4_t XTMP4, uint32x4_t XTMP5, int j)
{
    uint32_t t0, t1, t2, t3, t4, t5, T0, T1, T2, T3, T4, W;

    // Load and move operations using NEON
    W = vgetq_lane_u32(X0, 3);             // W[-13]
    t0 = rotate_right(A, 20);                     // A <<< 12
    X0 = veorq_u32(XTMP1, XTMP0);          // W[0],W[1],W[2],XXX
    t1 = B | C;                            // B | C
    t3 = Tj[j];          // Tj <<< j, assume loaded from a memory address
    j+=1;
    t2 = t0 + E;                          // (A <<< 12) + E
    T0 = vgetq_lane_u32(X0, 0);            // W[0]
    T3 = B & C;                            // B & C
    T4 = A & t1;                           // A & (B | C)
    T1 = vgetq_lane_u32(XTMP2, 3);         // W[-13] ^ W[-6]
    t5 = F ^ G;                           // F ^ G
    t1 = T3 | T4;                         // FF(A, B, C)
    T2 = vgetq_lane_u32(XTMP0, 3);         // (W[-10] <<< 7) ^ W[-3]
    T1 = rotate_right(T0, 17) ^ T1;               // Z = W[-13] ^ W[-6] ^ (W[0] <<< 15)
    t4 = t2 + t3;                         // (A <<< 12) + E + (Tj <<< j)
    T3 = rotate_right(T1, 17);                    // Z <<< 15
    t5 = t5 & E;                              // (F ^ G) & E
    H = H + W;                                // H + Wj
    t4 = rotate_right(t4, 25);                    // SS1
    T1 = T1 ^ rotate_right(T1, 9);                    // Z ^ (Z <<< 23)
    t5 = t5 ^ G;                               // GG(E, F, G)
    D = D + t1;                               // FF(A, B, C) + D
    B = rotate_right(B, 23);                      // B <<< 9
    W = vgetq_lane_u32(XFER, 3);           // WW[-13]
    t1 = t4 + t5;                          // GG(E, F, G) + SS1
    t2= t0 ^ t4;                          // SS2
    H = H + t1;                               // TT2 = GG(E, F, G) + H + SS1 + Wj
    T1 = T1 ^ T3;                              // Z ^ (Z <<< 15) ^ (Z <<< 23)
    F = rotate_right(F, 13);                      // F <<< 19
    t3 = rotate_right(H, 23);
    D = D + t2;                               // FF(A, B, C) + D + SS2
    H = H ^ rotate_right(H, 15);
    T2 = T1 ^ T2;                              // W[3]
    D = D + W;                                // TT1 = FF(A, B, C) + D + SS2 + W'j
    H = H ^ t3;                              // P0(TT2)
    X0=vsetq_lane_u32(T2, X0, 3);             // W[0],W[1],W[2],W[3]
}

void THIRD_12_ROUNDS_WITHOUT_SCHED_1(uint32x4_t X0, uint32x4_t X1, uint32x4_t X2, uint32x4_t X3, uint32_t A, uint32_t B, uint32_t C, uint32_t D,uint32_t E, uint32_t F, uint32_t G, uint32_t H,\
uint32x4_t XFER, int j)
{
    uint32_t t0, t1, t2, t3, t4, t5, W, T0, T1;

    // XOR the first two NEON registers
    XFER = veorq_u32(X0, X1);  // WW

    // Rotate and other bitwise operations
    t0 = rotate_right(A, 20);         // A <<< 12
    t1 = B | C;                // B | C
    T0 = B & C;                // B & C
    T1 = A & t1;               // A & (B | C)
    W = vgetq_lane_u32(X0, 0); // W[-16], taking first lane of X0

    // Assuming $TBL points to an array of uint32_t, we access it like this
    // This implies $TBL needs to be a globally accessible array
    t3 = Tj[j];               // Tj <<< j
    j+=1;
    // Addition and masking operations
    t2 = t0 + E;              // (A <<< 12) + E
    t5 = F ^ G;              // F ^ G
    t1 = T0 | T1;             // FF(A, B, C)

    // More calculations
    t4 = t2 + t3;             // (A <<< 12) + E + (Tj <<< j)
    t5 = t5 & E;                  // (F ^ G) & E
    H = H + W;                    // H + Wj
    t4 = rotate_right(t4, 25);        // SS1

    // Final round of operations
    t5 = t5 ^ G;                   // GG(E, F, G)
    D = D + t1;                   // FF(A, B, C) + D
    B = rotate_right(B, 23);          // B <<< 9
    t1 = t4 + t5;              // GG(E, F, G) + SS1
    W = vgetq_lane_u32(XFER, 0); // WW[-16], taking first lane of XFER
    t2 = t0 ^ t4;              // SS2

    // Updating H and D
    H = H + t1;                   // TT2 = GG(E, F, G) + H + SS1 + Wj
    F = rotate_right(F, 13);          // F <<< 19
    t3 = rotate_right(H, 23);
    D = D + t2;                   // FF(A, B, C) + D + SS2
    H = H ^ rotate_right(H, 15);
    D = D + W;                    // TT1 = FF(A, B, C) + D + SS2 + W'j
    H = H ^ t3;                  // P0(TT2)
}

void THIRD_12_ROUNDS_WITHOUT_SCHED_2(uint32x4_t X0, uint32x4_t X1, uint32x4_t X2, uint32x4_t X3, uint32_t A, uint32_t B, uint32_t C, uint32_t D,uint32_t E, uint32_t F, uint32_t G, uint32_t H,uint32x4_t XFER, int j)
{
    uint32_t t0, t1, t2, t3, t4, t5, W, T0, T1;

    // Bitwise operations
    t0 = rotate_right(A, 20);         // A <<< 12
    t1 = B | C;                // B | C
    T0 = B & C;                // B & C
    T1 = A & t1;               // A & (B | C)
    W = vgetq_lane_u32(X0, 1); // Extracting the second 32-bit integer from X0 vector

    t3 = Tj[j];               // Tj <<< j
    j+=1;
    // Additional computations
    t2 = t0 + E;              // (A <<< 12) + E
    t5 = F ^ G;               // F ^ G
    t1 = T0 | T1;             // FF(A, B, C)

    t4 = t2 + t3;             // (A <<< 12) + E + (Tj <<< j)
    t5 = t5 & E;                  // (F ^ G) & E
    H = H + W;                    // H + Wj
    t4 = rotate_right(t4, 25);        // SS1
    t5 = t5 ^ G;                   // GG(E, F, G)

    D = D + t1;                   // FF(A, B, C) + D
    B = rotate_right(B, 23);          // B <<< 9
    t1 = t4 + t5;              // GG(E, F, G) + SS1

    W = vgetq_lane_u32(XFER, 1); // Extracting the second 32-bit integer from XFER vector
    t2 = t0 ^ t4;              // SS2

    H = H + t1;                   // TT2 = GG(E, F, G) + H + SS1 + Wj
    F = rotate_right(F, 13);          // F <<< 19
    t3 = rotate_right(H, 23);

    D = D + t2;                   // FF(A, B, C) + D + SS2
    H = H ^ rotate_right(H, 15);
    D = D + W;                    // TT1 = FF(A, B, C) + D + SS2 + W'j
    H = H ^ t3;                  // P0(TT2)
}

void THIRD_12_ROUNDS_WITHOUT_SCHED_3(uint32x4_t X0, uint32x4_t X1, uint32x4_t X2, uint32x4_t X3,
                                     uint32_t A, uint32_t B, uint32_t C, uint32_t D,uint32_t E, uint32_t F, uint32_t G, uint32_t H,uint32x4_t XFER, int j) {
    uint32_t t0, t1, t2, t3, t4, t5, T0, T1, W;

    t0 = rotate_right(A, 20);                  // A <<< 12
    t1 = B | C;               // B | C
    t3 = Tj[j];
    j+=1;
    t2 = t0 + E;              // (A <<< 12) + E
    T0 = B & C;               // B & C
    T1 = A & t1;              // A & (B | C)
    W = vgetq_lane_u32(X0, 2);
    t5 = F ^ G;               // F ^ G
    t1 = T0 | T1;             // FF(A, B, C)
    t4 = t2 + t3;             // (A <<< 12) + E + (Tj <<< j)
    t5 = t5 & E;              // (F ^ G) & E
    H = H + W;                // H + Wj
    t4 = rotate_right(t4, 25);                 // SS1
    t5 = t5 ^ G;              // GG(E, F, G)
    D = t1 + D;               // FF(A, B, C) + D
    B = rotate_right(B, 23);                   // B <<< 9
    t1 = t4 + t5;             // GG(E, F, G) + SS1
    W = vgetq_lane_u32(XFER, 2); // Extracting the third 32-bit integer from XFER vector
    t2 = t0 ^ t4;             // SS2
    H = H + t1;               // TT2 = GG(E, F, G) + H + SS1 + Wj
    F = rotate_right(F, 13);                   // F <<< 19
    t3 = rotate_right(H, 23);
    D = D + t2;               // FF(A, B, C) + D + SS2
    H = H ^ rotate_right(H, 15);
    D = D + W;                // TT1 = FF(A, B, C) + D + SS2 + W'j
    H = H ^ t3;               // P0(TT2)
}

void THIRD_12_ROUNDS_WITHOUT_SCHED_4(uint32x4_t X0, uint32x4_t X1, uint32x4_t X2, uint32x4_t X3,
                                          uint32_t A, uint32_t B, uint32_t C, uint32_t D,uint32_t E, uint32_t F, uint32_t G, uint32_t H,uint32x4_t XFER, int j)
{
    uint32_t t0, t1, t2, t3, t4, t5;
    uint32_t T0, T1, W, WW;

    t0 = rotate_right(A, 20);              // A <<< 12
    t1 = B | C;                 // B | C
    t3 = Tj[j];  // Tj <<< j
    j+=1;
    t2 = t0 + E;                // (A <<< 12) + E
    T0 = B & C;                 // B & C
    T1 = A & t1;                // A & (B | C)
    W = vgetq_lane_u32(X0, 3); // W[-16]
    t5 = F ^ G;                 // F ^ G
    t1 = T0 | T1;               // FF(A, B, C)
    t4 = t2 + t3;               // (A <<< 12) + E + (Tj <<< j)
    t5 = t5 & E;                // (F ^ G) & E
    H = H + W;                  // H + Wj
    t4 = rotate_right(t4, 25);             // SS1
    t5 = t5 ^ G;                // GG(E, F, G)
    D = t1 + D;                 // FF(A, B, C) + D
    B = rotate_right(B, 23);               // B <<< 9
    t1 = t4 + t5;               // GG(E, F, G) + SS1
    W = vgetq_lane_u32(XFER, 3);  // WW[-16]
    t2 = t0 ^ t4;               // SS2
    H = H + t1;                 // TT2 = GG(E, F, G) + H + SS1 + Wj
    F = rotate_right(F, 13);               // F <<< 19
    t3 = rotate_right(H, 23);
    D = D + t2;                 // FF(A, B, C) + D + SS2
    H = H ^ rotate_right(H, 15);
    D = D + W;                // TT1 = FF(A, B, C) + D + SS2 + W'j
    H = H ^ t3;               // P0(TT2)
}

void sm3_compress_neon(uint32_t digest[8], const uint8_t *buf, uint64_t nb) {
    // 设置：将初始摘要加载到 NEON 寄存器
    u32 a,b,c,d,e,f,g,h;
    uint32_t A,B,C,D,E,F,G,H;
    uint32x4_t low = vld1q_u32(digest);
    uint32x4_t high = vld1q_u32(digest + 4);
    
    // 分别提取每个 uint32_t 值
    a = vgetq_lane_u32(low, 0);     
    b = vgetq_lane_u32(low, 1);
    c = vgetq_lane_u32(low, 2);
    d = vgetq_lane_u32(low, 3);
    e = vgetq_lane_u32(high, 0);
    f = vgetq_lane_u32(high, 1);
    g = vgetq_lane_u32(high, 2);
    h = vgetq_lane_u32(high, 3);
    A = a;
    B = b;
    C = c;
    D = d;
    E = e;
    F = f;
    G = g;
    F = h;


    // 主压缩循环
    while (nb--) {
        // 从数据块加载 64 字节到 NEON 寄存器
        uint8x16_t M0 = vld1q_u8(buf);
        uint8x16_t M1 = vld1q_u8(buf + 16);
        uint8x16_t M2 = vld1q_u8(buf + 32);
        uint8x16_t M3 = vld1q_u8(buf + 48);
        buf += 64; // 为下一次迭代移动块指针

        // 如有必要根据字节顺序翻转字节
        if (ENDIANESS==1){
        M0 = vrev32q_u8(M0);
        M1 = vrev32q_u8(M1);
        M2 = vrev32q_u8(M2);
        M3 = vrev32q_u8(M3);
        }
        int j=0;

        uint32x4_t X0=vreinterpretq_u32_u8(M0);
        uint32x4_t X1=vreinterpretq_u32_u8(M1);
        uint32x4_t X2=vreinterpretq_u32_u8(M2);
        uint32x4_t X3=vreinterpretq_u32_u8(M3);
        // 模拟轮次和消息计划更新
        // 需要定义这些函数以模仿汇编中的操作
        // 示例：
        // V1 = sm3_round(V1, V0, M0, ...);
        // V0 = sm3_schedule(V0, X0, X1, X2, X3, ...);
        uint32x4_t XFER; uint32x4_t XTMP0; uint32x4_t XTMP1;uint32x4_t XTMP2; uint32x4_t XTMP3; uint32x4_t XTMP4; uint32x4_t XTMP5;
        for (int i = 0; i < 4; i++) {
            FIRST_16_ROUNDS_AND_SCHED_1(X0, X1, X2, X3, A, B, C, D, E, F, G, H, XFER, XTMP0, XTMP1, XTMP2, XTMP3, XTMP4, XTMP5,j);
            FIRST_16_ROUNDS_AND_SCHED_2(X0, X1, X2, X3, D, A, B, C, H, E, F, G, XFER, XTMP0, XTMP1, XTMP2, XTMP3, XTMP4, XTMP5,j);
            FIRST_16_ROUNDS_AND_SCHED_3(X0, X1, X2, X3, C, D, A, B, G, H, E, F, XFER, XTMP0, XTMP1, XTMP2, XTMP3, XTMP4, XTMP5,j);
            FIRST_16_ROUNDS_AND_SCHED_4(X0, X1, X2, X3, B, C, D, A, F, G, H, E, XFER, XTMP0, XTMP1, XTMP2, XTMP3, XTMP4, XTMP5,j);
            circular_shift(X0, X1, X2, X3);
        }

        for (int i = 0; i < 9; i++) {
            SECOND_36_ROUNDS_AND_SCHED_1(X0, X1, X2, X3, A, B, C, D, E, F, G, H, XFER, XTMP0, XTMP1, XTMP2, XTMP3, XTMP4, XTMP5,j);
            SECOND_36_ROUNDS_AND_SCHED_2(X0, X1, X2, X3, D, A, B, C, H, E, F, G, XFER, XTMP0, XTMP1, XTMP2, XTMP3, XTMP4, XTMP5,j);
            SECOND_36_ROUNDS_AND_SCHED_3(X0, X1, X2, X3, C, D, A, B, G, H, E, F, XFER, XTMP0, XTMP1, XTMP2, XTMP3, XTMP4, XTMP5,j);
            SECOND_36_ROUNDS_AND_SCHED_4(X0, X1, X2, X3, B, C, D, A, F, G, H, E, XFER, XTMP0, XTMP1, XTMP2, XTMP3, XTMP4, XTMP5,j);
            circular_shift(X0, X1, X2, X3);
        }

        for (int i = 0; i < 3; i++) {
            THIRD_12_ROUNDS_WITHOUT_SCHED_1(X0, X1, X2, X3, A, B, C, D, E, F, G, H, XFER,j);
            THIRD_12_ROUNDS_WITHOUT_SCHED_2(X0, X1, X2, X3, D, A, B, C, H, E, F, G, XFER,j);
            THIRD_12_ROUNDS_WITHOUT_SCHED_3(X0, X1, X2, X3, C, D, A, B, G, H, E, F, XFER,j);
            THIRD_12_ROUNDS_WITHOUT_SCHED_4(X0, X1, X2, X3, B, C, D, A, F, G, H, E, XFER,j);
            circular_shift(X0, X1, X2, X3);
        }
        // 重复所需轮次和计划
            a = a ^ A;
            b = b ^ B;
            c = c ^ C;
            d = d ^ D;
            e = e ^ E;
            f = f ^ F;
            g = g ^ G;
            h = h ^ H;
            A=a;
            B=b;
            C=c;
            D=d;
            E=e;
            F=f;
            G=g;
            H=h;
    }

    // 完成处理：将计算的状态存回摘要

    // 压缩完成后保存结果
    digest[0]=a;
    digest[1]=b;
    digest[2]=c;
    digest[3]=d;
    digest[4]=e;
    digest[5]=f;
    digest[6]=g;
    digest[7]=h;
}

