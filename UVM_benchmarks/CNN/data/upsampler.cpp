#include <iostream>
#include <fstream>
#include <vector>
#include <stdint.h>

uint32_t swap_endian(uint32_t val) {
    return ((val << 24) & 0xff000000) |
           ((val << 8)  & 0x00ff0000) |
           ((val >> 8)  & 0x0000ff00) |
           ((val >> 24) & 0x000000ff);
}

void upscale(const std::string& in_img, const std::string& in_lbl, 
             const std::string& out_img, const std::string& out_lbl, 
             int repeats) {
    
    std::ifstream if_img(in_img, std::ios::binary);
    if (!if_img) { std::cerr << "Can't find input image file!\n"; return; }

    uint32_t magic, count, rows, cols;
    if_img.read((char*)&magic, 4);
    if_img.read((char*)&count, 4);
    if_img.read((char*)&rows, 4);
    if_img.read((char*)&cols, 4);

    uint32_t original_count = swap_endian(count);
    uint32_t new_count_swapped = swap_endian(original_count * repeats);

    // Read all pixel data into memory once
    std::vector<char> img_buffer((std::istreambuf_iterator<char>(if_img)), 
                                  std::istreambuf_iterator<char>());
    if_img.close();

    std::ofstream of_img(out_img, std::ios::binary);
    of_img.write((char*)&magic, 4);
    of_img.write((char*)&new_count_swapped, 4);
    of_img.write((char*)&rows, 4);
    of_img.write((char*)&cols, 4);

    for (int i = 0; i < repeats; ++i) {
        of_img.write(img_buffer.data(), img_buffer.size());
    }
    of_img.close();

    // 2. Process Labels
    std::ifstream if_lbl(in_lbl, std::ios::binary);
    if_lbl.read((char*)&magic, 4);
    if_lbl.read((char*)&count, 4);
    
    std::vector<char> lbl_buffer((std::istreambuf_iterator<char>(if_lbl)), 
                                  std::istreambuf_iterator<char>());
    if_lbl.close();

    std::ofstream of_lbl(out_lbl, std::ios::binary);
    of_lbl.write((char*)&magic, 4);
    of_lbl.write((char*)&new_count_swapped, 4);

    for (int i = 0; i < repeats; ++i) {
        of_lbl.write(lbl_buffer.data(), lbl_buffer.size());
    }
    of_lbl.close();

    std::cout << "Successfully created massive files with " 
              << original_count * repeats << " images.\n";
}

int main() {
    upscale("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
            "data/massive-images.idx3-ubyte", "data/massive-labels.idx1-ubyte", 
            65);
    return 0;
}