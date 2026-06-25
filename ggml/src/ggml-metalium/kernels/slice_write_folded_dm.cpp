#include <cstdint>
// Copies the WKV7 region-2 final state directly into the row-folded state
// cache, replacing the host-side flatten -> row_fold -> slice_write path.
//
// This kernel only handles the validated S==H==64, G==1 case. In that layout,
// each head contributes exactly one source row, and the data can be copied
// face-for-face with no reshuffle.
//
// Layout summary:
//   Source: wkv_output [1,1,T+S,C], TILE bf16, where C = S*H = 4096
//     For head h, state (i,j) is at:
//       row = T + h
//       col = i*64 + j
//
//   Dest: folded cache [1,1,C*S/32,32], TILE bf16
//     For head h, state (i,j) is at:
//       row = h*128 + 2*i + j/32
//       col = j%32
//
// Grouping columns into 32-wide chunks (g = col/32):
//   source row (T+h), cols [g*32, g*32+32)
//     maps directly to
//   folded row (h*ng + g), cols [0,32)
//
// So each head's 4096-wide source row becomes a [128,32] folded block, and the
// kernel only moves 32-byte face rows from source to destination.
//
// TT tile layout:
//   A tile contains four 16x16 faces at byte offsets:
//     TL=0, TR=512, BL=1024, BR=1536
//   Logical row r within a tile occupies two 32-byte face rows at:
//     (r/16)*1024 + (r%16)*32
//   and
//     (r/16)*1024 + (r%16)*32 + 512
//
// Runtime args:
//   0 T          : region-2 row offset (n_tokens)
//   1 H          : number of heads (64 here)
//   2 ng         : 32-column groups per head = C/32
//   3 src_addr   : DRAM base of wkv_output
//   4 dst_addr   : DRAM base of the folded cache
//   5 inst_start : first head handled by this core (inclusive)
//   6 inst_end   : one past the last head handled by this core
//
// Compile args: src TensorAccessorArgs, then dst TensorAccessorArgs.

void kernel_main() {
    uint32_t T          = get_arg_val<uint32_t>(0);
    uint32_t H          = get_arg_val<uint32_t>(1);
    uint32_t ng         = get_arg_val<uint32_t>(2);
    uint32_t src_addr   = get_arg_val<uint32_t>(3);
    uint32_t dst_addr   = get_arg_val<uint32_t>(4);
    uint32_t inst_start = get_arg_val<uint32_t>(5);
    uint32_t inst_end   = get_arg_val<uint32_t>(6);
    (void)H;

    const uint32_t tb = get_tile_size(0);   // bf16 32x32 = 2048B (src/dst page size)
    constexpr uint32_t SRC_NA = TensorAccessorArgs<0>::num_compile_time_args();
    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr auto dst_args = TensorAccessorArgs<SRC_NA>();
    const auto src_acc = TensorAccessor(src_args, src_addr, tb);
    const auto dst_acc = TensorAccessor(dst_args, dst_addr, tb);

    // L1 relay scratch (CB 0): one head = ng groups * 2 faces * 32B.
    const uint32_t l1 = get_write_ptr(0);
    const uint32_t dtiles = ng / 32;        // dest folded tiles per head

    for (uint32_t h = inst_start; h < inst_end; h++) {
        const uint32_t R   = T + h;
        const uint32_t sr  = R % 32;
        const uint32_t str = R / 32;
        const uint32_t so0 = (sr / 16) * 1024 + (sr % 16) * 32;  // src face (sr/16, 0)
        const uint32_t so1 = so0 + 512;                          // src face (sr/16, 1)

        // Read this head's whole source row (all ng col-tiles, both faces) into L1.
        for (uint32_t g = 0; g < ng; g++) {
            const uint32_t src_page = str * ng + g;
            noc_async_read(src_acc.get_noc_addr(src_page, so0), l1 + g * 64 + 0,  32);
            noc_async_read(src_acc.get_noc_addr(src_page, so1), l1 + g * 64 + 32, 32);
        }
        noc_async_read_barrier();

        // Write each group verbatim to its folded row (head h -> tiles [4h, 4h+dtiles)).
        for (uint32_t g = 0; g < ng; g++) {
            const uint32_t dst_page = h * dtiles + g / 32;
            const uint32_t srd = g % 32;
            const uint32_t do0 = (srd / 16) * 1024 + (srd % 16) * 32;
            const uint32_t do1 = do0 + 512;
            noc_async_write(l1 + g * 64 + 0,  dst_acc.get_noc_addr(dst_page, do0), 32);
            noc_async_write(l1 + g * 64 + 32, dst_acc.get_noc_addr(dst_page, do1), 32);
        }
        noc_async_write_barrier();
    }
}
