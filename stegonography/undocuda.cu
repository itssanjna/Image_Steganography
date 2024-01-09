#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <png.h>
#include <openssl/sha.h>

__global__ void modifyPNGKernel(png_bytep *rowPointers, int width, int height, int passcode, char *message, int stringlen)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    SHA256_CTX shaContext;
    SHA256_Init(&shaContext);

    if (x < width && y < height)
    {
        int i = y * width + x;
        if (i < stringlen)
        {
            int num = passcode + i;
            int hash = 0;
            int multiplier = 1;
            while (num > 0)
            {
                int digit = num % 10;
                hash += digit * multiplier;
                num /= 10;
                multiplier *= 10;
            }
            SHA256((const unsigned int)passcode, 4 * sizeof(int), (unsigned int)hash);
            if (i == hash[i])
            {
                png_bytep row = rowPointers[y];
                png_bytep px = &(row[x * 4]); // Assuming RGBA color format
                (message[i / 8]) += px[hash % 4 + (i % 8)] & 0x1;
                message[i / 8] = message[i / 8] << (i % 8);
            }
        }
    }
}

void modifyPNG(const char *inputPath, const char *outputPath, char *message, int passcode, int stringlen)
{

    FILE *inputFile = fopen(inputPath, "rb");
    if (!inputFile)
    {
        printf("Failed to open the input PNG file\n");
        return;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png)
    {
        fclose(inputFile);
        printf("Failed to create PNG read struct\n");
        return;
    }

    png_infop info = png_create_info_struct(png);
    if (!info)
    {
        png_destroy_read_struct(&png, NULL, NULL);
        fclose(inputFile);
        printf("Failed to create PNG info struct\n");
        return;
    }

    if (setjmp(png_jmpbuf(png)))
    {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(inputFile);
        printf("Failed to setjmp for PNG file\n");
        return;
    }

    png_init_io(png, inputFile);
    png_read_info(png, info);

    int width = png_get_image_width(png, info);
    int height = png_get_image_height(png, info);
    png_byte colorType = png_get_color_type(png, info);
    png_byte bitDepth = png_get_bit_depth(png, info);

    if (bitDepth == 16)
        png_set_strip_16(png);

    if (colorType == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);

    if (colorType == PNG_COLOR_TYPE_GRAY && bitDepth < 8)
        png_set_expand_gray_1_2_4_to_8(png);

    if (png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);

    if (colorType == PNG_COLOR_TYPE_RGB ||
        colorType == PNG_COLOR_TYPE_GRAY ||
        colorType == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

    if (colorType == PNG_COLOR_TYPE_GRAY ||
        colorType == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);
    png_bytep *rowPointers = (png_bytep *)malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++)
    {
        rowPointers[y] = (png_byte *)malloc(png_get_rowbytes(png, info));
    }

    png_read_image(png, rowPointers);

    png_destroy_read_struct(&png, &info, NULL);
    fclose(inputFile);
    // Allocate memory on the GPU

    png_bytep *d_rowPointers;

    cudaMalloc((void **)&d_rowPointers, sizeof(png_bytep) * height);

    size_t rowSize = png_get_rowbytes(png, info);
    png_bytep d_imageData;
    cudaMalloc((void **)&d_imageData, rowSize * height);
    for (int y = 0; y < height; y++)
    {
        cudaMemcpy(d_imageData + y * rowSize, rowPointers[y], rowSize, cudaMemcpyHostToDevice);
    }

    char *d_message;
    cudaMalloc((void **)&d_message, sizeof(char) * (stringlen + 1));
    cudaMemcpy(d_message, message, sizeof(char) * (stringlen + 1), cudaMemcpyHostToDevice);
    // Launch the kernel
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    modifyPNGKernel<<<gridSize, blockSize>>>(d_rowPointers, width, height, passcode, d_message, stringlen);
    // Copy the result back to the host
    cudaMemcpy(message, d_message, sizeof(char) * (stringlen + 1), cudaMemcpyDeviceToHost);
    message[strlen(message)] = '\0';
    printf("\nmessage: %s\n", message);
    // Free memory on the GPU
    for (int y = 0; y < height; y++)
    {
        cudaFree(d_rowPointers[y]);
    }
    cudaFree(d_rowPointers);
    cudaFree(d_message);

    // Save the modified PNG file
    FILE *outputFile = fopen(outputPath, "wb");
    if (!outputFile)
    {
        printf("Failed to open the output PNG file\n");
        for (int y = 0; y < height; y++)
        {
            free(rowPointers[y]);
        }
        free(rowPointers);
        return;
    }

    png_structp pngOutput = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!pngOutput)
    {
        fclose(outputFile);
        printf("Failed to create PNG write struct\n");
        for (int y = 0; y < height; y++)
        {
            free(rowPointers[y]);
        }
        free(rowPointers);
        return;
    }

    png_infop infoOutput = png_create_info_struct(pngOutput);
    if (!infoOutput)
    {
        fclose(outputFile);
        printf("Failed to create PNG info struct\n");
        png_destroy_write_struct(&pngOutput, NULL);
        for (int y = 0; y < height; y++)
        {
            free(rowPointers[y]);
        }
        free(rowPointers);
        return;
    }

    if (setjmp(png_jmpbuf(pngOutput)))
    {
        fclose(outputFile);
        printf("Failed to setjmp for output PNG file\n");
        png_destroy_write_struct(&pngOutput, &infoOutput);
        for (int y = 0; y < height; y++)
        {
            free(rowPointers[y]);
        }
        free(rowPointers);
        return;
    }

    png_init_io(pngOutput, outputFile);

    png_set_IHDR(pngOutput, infoOutput, width, height, bitDepth, PNG_COLOR_TYPE_RGBA,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(pngOutput, infoOutput);
    png_write_image(pngOutput, rowPointers);
    png_write_end(pngOutput, NULL);

    fclose(outputFile);

    png_destroy_write_struct(&pngOutput, &infoOutput);

    for (int y = 0; y < height; y++)
    {
        free(rowPointers[y]);
    }
    free(rowPointers);
}

int main()
{
    const char *inputPath = "newimage.png";
    const char *outputPath = "newimage.png";
    int passcode;
    printf("Enter passcode-1: ");
    scanf("%d", &passcode);

    printf("Enter passcode-2: ");
    unsigned int maxLength;
    scanf("%d", &maxLength);
    maxLength = (maxLength - 4432) / 2;

    char *message = (char *)malloc(sizeof(char) * (maxLength + 1));

    modifyPNG(inputPath, outputPath, message, passcode, maxLength);
    printf("\nThe embedded message was: %s\n", message);

    free(message);
    return 0;
}
