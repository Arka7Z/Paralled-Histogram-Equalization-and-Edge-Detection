				#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <png.h>

#define HI(num) (((num) & 0x0000FF00) << 8) 
#define LO(num) ((num) & 0x000000FF) 

typedef struct _PGMData
{
    int row;
    int col;
    int max_gray;
    int **matrix;
} PGMData;

void SkipComments(FILE *fp)
{
    int ch;
    char line[100];
    while ((ch = fgetc(fp)) != EOF && isspace(ch)) {
        ;
    }
 
    if (ch == '#')
    {
        fgets(line, sizeof(line), fp);
        SkipComments(fp);
    } else
    {
        fseek(fp, -1, SEEK_CUR);
    }
} 

void writePGM(const char *filename, const int *image, int row, int col, int maxGray)
{
    FILE *pgmFile;
    int i, j;
    int hi, lo;
 
    pgmFile = fopen(filename, "wb");
    if (pgmFile == NULL) {
        perror("cannot open file to write");
        exit(EXIT_FAILURE);
    }
 
    fprintf(pgmFile, "P5\n");
    fprintf(pgmFile, "%d %d\n", col, row);
    fprintf(pgmFile, "%d\n", maxGray);
 
    if (maxGray > 255) 
    {
        for (int k = 0; k < row * col; k++)
        {
            int i = k / col, j = k % col;
            hi = HI(image[k]);
            lo = LO(image[k]);
            fputc(hi, pgmFile);
            fputc(lo, pgmFile);
        }
    }
    else 
    {

        for (int k = 0; k < row * col; k++)
        {
            int i = k / col, j = k % col;
            lo = LO(image[k]);
            fputc(lo, pgmFile);
        }
    }
 
    fclose(pgmFile);
}

int main(int argc, char *argv[])
{
    int size, rank;
    int row, col, maxGray = 255;
    int* globalImg;

    MPI_Init(&argc,&argv),
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        /* Reading png file in master node */

        int width, height;		// height: ncols, widht: nrows
        png_byte color_type;
        png_byte bit_depth;
        png_bytep *row_pointers;
        char *filename = "input.png";
        if (argc > 1)
        	filename = argv[1];
        FILE *fp = fopen(filename, "rb");
        if(!fp) abort();
        png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        if(!png) abort();
        png_infop info = png_create_info_struct(png);
        if(!info) abort();
        if(setjmp(png_jmpbuf(png))) abort();
        png_init_io(png, fp);
        png_read_info(png, info);
        width      = png_get_image_width(png, info);
        height     = png_get_image_height(png, info);
        color_type = png_get_color_type(png, info);
        bit_depth  = png_get_bit_depth(png, info);

        if(bit_depth == 16)
            png_set_strip_16(png);

        if(color_type == PNG_COLOR_TYPE_PALETTE)
            png_set_palette_to_rgb(png);

        if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
            png_set_expand_gray_1_2_4_to_8(png);

        if(png_get_valid(png, info, PNG_INFO_tRNS))
            png_set_tRNS_to_alpha(png);

        if(color_type == PNG_COLOR_TYPE_RGB ||
            color_type == PNG_COLOR_TYPE_GRAY ||
            color_type == PNG_COLOR_TYPE_PALETTE)
            png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

          if(color_type == PNG_COLOR_TYPE_GRAY ||
		     color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
		    png_set_gray_to_rgb(png);

        png_read_update_info(png, info);

        row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
        for(int y = 0; y < height; y++) {
            row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png,info));
        }
        png_read_image(png, row_pointers);
        fclose(fp);

        int *img = (int*)malloc(height * width * sizeof(int));
        globalImg = (int*)malloc(height * width * sizeof(int));
        for(int y = 0; y < height; y++) 
        {
            png_bytep row = row_pointers[y];
            for(int x = 0; x < width; x++) 
            {
                png_bytep px = &(row[x * 4]);
                int val =  round(0.3 * px[0] + 0.59 * px[1] + 0.11 * px[2]);
                img[x * height + y] = val;
            }
        }

        for (int i = 0; i < width; i++)
            for (int j = 0; j < height; j++)
                globalImg[j * width + i] = img[i * height + j];
        row = height;
        col = width;
        maxGray = 255;        
    }

    /* Broadcast the image parameters i.e. maxGray, row size and column size*/

    MPI_Bcast( &maxGray, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast( &row, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast( &col, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    int* localHistogram = (int*)malloc((maxGray + 1) * sizeof(int));
    int* globalHistogram = (int*)malloc((maxGray + 1) * sizeof(int));
    for (int i = 0; i <= maxGray; i++)
        localHistogram[i] = 0;

    /* Scatter(scatterv) the image among the processes */

    int *sendcounts = (int*)malloc(sizeof(int)*size);
    int *displs = (int*)malloc(sizeof(int)*size);
    int rem = (row * col) % size;
    int sum = 0;

    for (int i = 0; i < size; i++) {
        sendcounts[i] = (row * col) / size;
        if (rem > 0) {
            sendcounts[i]++;
            rem--;
        }

        displs[i] = sum;
        sum += sendcounts[i];
    }
    int* localImg = (int*)malloc(sendcounts[rank] * sizeof(int));
    MPI_Scatterv(globalImg, sendcounts, displs, MPI_INT, localImg, sendcounts[rank], MPI_INT, 0, MPI_COMM_WORLD);  // each process receives sendcounts[rank] number of elements

    /* compute frequency of the local image in each proces */

    for (int i = 0; i < sendcounts[rank]; i++)
        localHistogram[localImg[i]]++;

    /* get hisogram of global image in each process */

    MPI_Allreduce(localHistogram, globalHistogram, maxGray + 1, MPI_INT, MPI_SUM,  MPI_COMM_WORLD);

    /* compute transformation matrix for the pixels i.e. transMat[p] = new value of pixel p */

    int totalPixels = row * col, curr = 0;
    int* transMat = (int*)malloc((maxGray + 1) * sizeof(int));
    for (int i = 0; i <= maxGray; i++) 
    { 
        curr += globalHistogram[i]; 
        transMat[i] = round((((float)curr) * maxGray) / totalPixels); 
    } 

    /* transform the local images in each process */
    for (int i = 0; i < sendcounts[rank]; i++)
        localImg[i] = transMat[localImg[i]];

    // MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    //             void *recvbuf, const int *recvcounts, const int *displs,
    //             MPI_Datatype recvtype, int root, MPI_Comm comm)

    
    MPI_Gatherv(localImg, sendcounts[rank], MPI_INT, globalImg, sendcounts, displs,  MPI_INT,  0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        writePGM("histeql.pgm", globalImg, row, col, maxGray);
    }

    /* Apply sobel operator */

    // sendcounts[i]: number of rows received by ith process
    rem = row % size, sum = 0;
    for (int i = 0; i < size; i++) 
    {
        sendcounts[i] = (row ) / size;
        if (rem > 0) {
            sendcounts[i]++;
            rem--;
        }

        displs[i] = sum;
        sum += sendcounts[i];
    }

    int tmpSendCounts[size];
    for (int i = 0; i < size; i++)
    {
        tmpSendCounts[i] = sendcounts[i] * col;
        if (i == 0 || i == size - 1)
            sendcounts[i]++;
        else
        sendcounts[i] += 2;        
    }
    for (int i = 0; i < size; i++)
        sendcounts[i] *= col;
    sum = 0;
    for (int i = 0; i < size; i++) 
    {
        displs[i] = sum;
        sum += tmpSendCounts[i];
    }
    for (int i = 1; i < size; i++)
        displs[i] -= col;

    int* localImgSobel = (int*)malloc(sendcounts[rank] * sizeof(int));
    MPI_Scatterv(globalImg, sendcounts, displs, MPI_INT, localImgSobel, sendcounts[rank], MPI_INT, 0, MPI_COMM_WORLD);
    
    int* localSobelResult = (int*)malloc(tmpSendCounts[rank] * sizeof(int));
    // tmpSendCounts[rank] = actual rows in op image of the process * col

    float sobelX[3][3] = { { -1, 0, 1 },
                          { -2, 0, 2 },
                          { -1, 0, 1 } };

    float sobelY[3][3] = { { -1, -2, -1 },
                         { 0,  0,  0 },
                         { 1,  2,  1 } };
    
    if (rank == 0)
    {
        for (int j = 0; j < col; j++)
            localSobelResult[j] = localImgSobel[j];
        for (int i = 1; i < tmpSendCounts[0] / col; i++)
            for (int j = 0; j < col; j++)
        {
            int sum; 
            float sumx = 0, sumy = 0;
            if ( j == 0 || j == (col - 1))
                sum = (sum > 255? 255 : (sum < 0? 0: sum));
            else
            {
                for (int r = -1; r <= 1; r++)
                    for (int c = -1; c <= 1; c++)
                    {
                        sumx += localImgSobel[(i + r) * col + j + c] * sobelX[r + 1][c + 1];
                        sumy += localImgSobel[(i + r) * col + j + c] * sobelY[r + 1][c + 1];;
                    }
                sum = (int)(sqrt(sumx * sumx + sumy * sumy));
                sum = (sum > 255? 255 : (sum < 0? 0: sum));
            }
            localSobelResult[i * col + j] = sum;
        }
    }
    if (rank > 0 && rank < size - 1)
    {
        for (int i = 1; i <= tmpSendCounts[rank] / col; i++)
            for (int j = 0; j < col; j++)
        {
            int sum; 
            float sumx = 0, sumy = 0;
            if ( i ==0 || i == row - 1 || j == 0 || j == (col - 1))
                sum = (sum > 255? 255 : (sum < 0? 0: sum));
            else
            {
                for (int r = -1; r <= 1; r++)
                    for (int c = -1; c <= 1; c++)
                    {
                        sumx += localImgSobel[(i + r) * col + j + c] * sobelX[r + 1][c + 1];
                        sumy += localImgSobel[(i + r) * col + j + c] * sobelY[r + 1][c + 1];;
                    }
                sum = (int)(sqrt(sumx * sumx + sumy * sumy));
                sum = (sum > 255? 255 : (sum < 0? 0: sum));
            }
            localSobelResult[(i - 1) * col + j] = sum;
        }
    }
    if (rank == size - 1)
    {
        for (int i = 1; i < tmpSendCounts[rank] / col - 1; i++)
            for (int j = 0; j < col; j++)
        {
            int sum; 
            float sumx = 0, sumy = 0;
            if ( i ==0 || i == row - 1 || j == 0 || j == (col - 1))
                sum = (sum > 255? 255 : (sum < 0? 0: sum));
            else
            {
                for (int r = -1; r <= 1; r++)
                    for (int c = -1; c <= 1; c++)
                    {
                        sumx += localImgSobel[(i + r) * col + j + c] * sobelX[r + 1][c + 1];
                        sumy += localImgSobel[(i + r) * col + j + c] * sobelY[r + 1][c + 1];;
                    }
                sum = (int)(sqrt(sumx * sumx + sumy * sumy));
                sum = (sum > 255? 255 : (sum < 0? 0: sum));
            }
            localSobelResult[(i - 1) * col + j] = sum;
        }
        for (int j = 0; j < col; j++)
            localSobelResult[(tmpSendCounts[rank] / col - 1) * col + j] = localImgSobel[tmpSendCounts[rank] + j];
    }

    for (int i = displs[rank]; i < displs[rank] + sendcounts[rank] && i < row; i++)
        for (int j = 0; j < col; j++)
    {
        int sum; 
        float sumx = 0, sumy = 0;
        if (i == 0 || i == (row - 1) || j == 0 || j == (col - 1))
            sum = (sum > 255? 255 : (sum < 0? 0: sum));
        else
        {
            for (int r = -1; r <= 1; r++)
                for (int c = -1; c <= 1; c++)
                {
                    sumx += localImgSobel[(i + r) * col + j + c] * sobelX[r + 1][c + 1];
                    sumy += localImgSobel[(i + r) * col + j + c] * sobelY[r + 1][c + 1];;
                }
            sum = (int)(sqrt(sumx * sumx + sumy * sumy));
            sum = (sum > 255? 255 : (sum < 0? 0: sum));
        }
        localSobelResult[(i - displs[rank]) * col + j] = sum;
        
    }

    sum = 0;
    for (int i = 0; i < size; i++) 
    {
        displs[i] = sum;
        sum += tmpSendCounts[i];
    }


    MPI_Gatherv(localSobelResult, tmpSendCounts[rank], MPI_INT, globalImg, tmpSendCounts, displs,  MPI_INT,  0, MPI_COMM_WORLD);

    if (rank == 0)
        writePGM("final.pgm", globalImg, row, col, maxGray);
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    
}
 
