#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define VERBOSE 1
#define BOOSTBLURFACTOR 90.0
#define THREADS 256
#define BLOCKS 1

int read_pgm_image(char *infilename, unsigned char **image, int *rows,
    int *cols);
int write_pgm_image(char *outfilename, unsigned char *image, int rows,
    int cols, char *comment, int maxval);

void canny(unsigned char *image, int rows, int cols, float sigma,
    float tlow, float thigh, unsigned char **edge, char *fname);
void gaussian_smooth(unsigned char *image, int rows, int cols, float sigma,
    short int **smoothedim);
void make_gaussian_kernel(float sigma, float **kernel, int *windowsize);
void derrivative_x_y(short int *smoothedim, int rows, int cols,
    short int **delta_x, short int **delta_y);
void magnitude_x_y(short int *delta_x, short int *delta_y, int rows, int cols,
    short int **magnitude);
void apply_hysteresis(short int *mag, unsigned char *nms, int rows, int cols,
    float tlow, float thigh, unsigned char *edge);
// void radian_direction(short int *delta_x, short int *delta_y, int rows,
//     int cols, float **dir_radians, int xdirtag, int ydirtag);
// double angle_radians(double x, double y);
    void non_max_supp(short *mag, short *gradx, short *grady, int nrows, int ncols,
unsigned char *result);

// Device declarations
void radian_direction(short int *delta_x, short int *delta_y, int rows,
    int cols, float **dir_radians, int xdirtag, int ydirtag);
__device__
double angle_radians(double x, double y);
__global__
void radian_direction_Device(short int *d_delta_x, short int *d_delta_y, int rows,
    int cols, float *d_dir_radians, int d_xdirtag, int d_ydirtag);
__global__
void gaussian_smooth_Device(int rows, int cols, int center, unsigned char *d_image,
    float *d_kernel, float *d_tempim, short int *d_smoothedim);
__global__
void derrivative_Device(int rows, int cols, short int *d_delta_x,
  short int *d_delta_y, short int *d_smoothedim);
__global__
void derrivative_x(int rows, int cols, short int *d_delta_x,
    short int *d_smoothedim);
__global__
void derrivative_y(int rows, int cols, short int *d_delta_y,
    short int *d_smoothedim);
__global__
void magnitude_x_y_Device(short int *delta_x, short int *delta_y, int rows, int cols,
    short int *magnitude);
    __global__
void non_max_supp_device(int nrows, int ncols, short *d_mag, short *d_gradx,
    short *d_grady, unsigned char *d_result);
__global__
void apply_hysteresis_edge(int rows, int cols, unsigned char *nms,
    unsigned char *edge, int *hist, short int *mag);


// Declaracion de Variables CUDA
short int *d_smoothedim;
short int *d_magnitude;
unsigned char *d_result;
short int *d_delta_x;
short int *d_delta_y;

int main(int argc, char *argv[])
{
    char *infilename = NULL;  /* Name of the input image */
    char *dirfilename = NULL; /* Name of the output gradient direction image */
    char outfilename[128];    /* Name of the output "edge" image */
    char composedfname[128];  /* Name of the output "direction" image */
    unsigned char *image;     /* The input image */
    unsigned char *edge;      /* The output edge image */
    int rows, cols;           /* The dimensions of the image. */
    float sigma,              /* Standard deviation of the gaussian kernel. */
    tlow,               /* Fraction of the high threshold in hysteresis. */
    thigh;              /* High hysteresis threshold control. The actual
    threshold is the (100 * thigh) percentage point
    in the histogram of the magnitude of the
    gradient image that passes non-maximal
    suppression. */

    cudaEvent_t start, end;
    float time;
    cudaEventCreate (&start);
    cudaEventCreate (&end);
    cudaEventRecord (start);
    printf("********************************************************\n");
    printf("*********************** CANNY CUDA *********************\n");
    printf("********************************************************\n");
    /****************************************************************************
    * Get the command line arguments.
    ****************************************************************************/
    if(argc < 5){
        fprintf(stderr,"\n<USAGE> %s image sigma tlow thigh [writedirim]\n",argv[0]);
        fprintf(stderr,"\n      image:      An image to process. Must be in ");
        fprintf(stderr,"PGM format.\n");
        fprintf(stderr,"      sigma:      Standard deviation of the gaussian");
        fprintf(stderr," blur kernel.\n");
        fprintf(stderr,"      tlow:       Fraction (0.0-1.0) of the high ");
        fprintf(stderr,"edge strength threshold.\n");
        fprintf(stderr,"      thigh:      Fraction (0.0-1.0) of the distribution");
        fprintf(stderr," of non-zero edge\n                  strengths for ");
        fprintf(stderr,"hysteresis. The fraction is used to compute\n");
        fprintf(stderr,"                  the high edge strength threshold.\n");
        fprintf(stderr,"      writedirim: Optional argument to output ");
        fprintf(stderr,"a floating point");
        fprintf(stderr," direction image.\n\n");
        exit(1);
    }

    infilename = argv[1];
    sigma = atof(argv[2]);
    tlow = atof(argv[3]);
    thigh = atof(argv[4]);

    if(argc == 6) dirfilename = infilename;
    else dirfilename = NULL;

    /****************************************************************************
    * Read in the image. This read function allocates memory for the image.
    ****************************************************************************/
    if(VERBOSE) printf("Reading the image %s.\n", infilename);
    if(read_pgm_image(infilename, &image, &rows, &cols) == 0){
        fprintf(stderr, "Error reading the input image, %s.\n", infilename);
        exit(1);
    }

    /****************************************************************************
    * Perform the edge detection. All of the work takes place here.
    ****************************************************************************/
    if(VERBOSE) printf("Starting Canny edge detection.\n");
    if(dirfilename != NULL){
        sprintf(composedfname, "%s_s_%3.2f_l_%3.2f_h_%3.2f.fim", infilename,
        sigma, tlow, thigh);
        dirfilename = composedfname;
    }
    canny(image, rows, cols, sigma, tlow, thigh, &edge, dirfilename);

    /****************************************************************************
    * Write out the edge image to a file.
    ****************************************************************************/
    sprintf(outfilename, "./out/imagen-cuda.pgm");
    if(VERBOSE) printf("Writing the edge iname in the file %s.\n", outfilename);

    char comment[4] = {""};
    if(write_pgm_image(outfilename, edge, rows, cols, comment, 255) == 0){
        fprintf(stderr, "Error writing the edge image, %s.\n", outfilename);
        exit(1);
    }
    free(image);
    cudaEventRecord (end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime (&time, start, end);
    printf("\nTiempo total del programa:\t\t%.3g segundos\n", time/1000);
    //---------------------------------------------------------------------
    FILE *archivo;
    char linea[6];
    int tiempoSerial = 0;
    int tiempo = (time/1000)*100000;
    archivo = fopen("./out/tiempoSerial","r");

    if (archivo == NULL)
    exit(1);
    else{
        while (feof(archivo) == 0){
            fgets(linea,100,archivo);
            tiempoSerial = atoi(linea);
        }
    }
    fclose(archivo);

    printf("speedup:\t\t\t\t%.3f",(float)tiempoSerial/tiempo);
    //---------------------------------------------------------------------
    printf("\n========================================================\n");
    return 0;
}

/*******************************************************************************
* PROCEDURE: canny
* PURPOSE: To perform canny edge detection.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void canny(unsigned char *image, int rows, int cols, float sigma,
    float tlow, float thigh, unsigned char **edge, char *fname)
{
    FILE *fpdir=NULL;          /* File to write the gradient image to.     */
    unsigned char *nms;        /* Points that are local maximal magnitude. */
    short int *smoothedim,     /* The image after gaussian smoothing.      */
    *delta_x,        /* The first devivative image, x-direction. */
    *delta_y,        /* The first derivative image, y-direction. */
    *magnitude;      /* The magnitude of the gadient image.      */
    float *dir_radians=NULL;   /* Gradient direction image.                */

    /****************************************************************************
    * Perform gaussian smoothing on the image using the input standard
    * deviation.
    ****************************************************************************/
    if(VERBOSE) printf("Smoothing the image using a gaussian kernel.\n");
    gaussian_smooth(image, rows, cols, sigma, &smoothedim);

    /****************************************************************************
    * Compute the first derivative in the x and y directions.
    ****************************************************************************/
    if(VERBOSE) printf("Computing the X and Y first derivatives.\n");
    derrivative_x_y(smoothedim, rows, cols, &delta_x, &delta_y);

    /****************************************************************************
    * This option to write out the direction of the edge gradient was added
    * to make the information available for computing an edge quality figure
    * of merit.
    ****************************************************************************/
    if(fname != NULL){
        /*************************************************************************
        * Compute the direction up the gradient, in radians that are
        * specified counteclockwise from the positive x-axis.
        *************************************************************************/
        radian_direction(delta_x, delta_y, rows, cols, &dir_radians, -1, -1);

        /*************************************************************************
        * Write the gradient direction image out to a file.
        *************************************************************************/
        if((fpdir = fopen(fname, "wb")) == NULL){
            fprintf(stderr, "Error opening the file %s for writing.\n", fname);
            exit(1);
        }
        fwrite(dir_radians, sizeof(float), rows*cols, fpdir);
        fclose(fpdir);
        free(dir_radians);
    }

    /****************************************************************************
    * Compute the magnitude of the gradient.
    ****************************************************************************/
    if(VERBOSE) printf("Computing the magnitude of the gradient.\n");
    magnitude_x_y(delta_x, delta_y, rows, cols, &magnitude);

    /****************************************************************************
    * Perform non-maximal suppression.
    ****************************************************************************/
    if(VERBOSE) printf("Doing the non-maximal suppression.\n");
    if((nms = (unsigned char *) calloc(rows*cols,sizeof(unsigned char)))==NULL){
        fprintf(stderr, "Error allocating the nms image.\n");
        exit(1);
    }
    non_max_supp(magnitude, delta_x, delta_y, rows, cols, nms);

    /****************************************************************************
    * Use hysteresis to mark the edge pixels.
    ****************************************************************************/
    if(VERBOSE) printf("Doing hysteresis thresholding.\n");
    if((*edge=(unsigned char *)calloc(rows*cols,sizeof(unsigned char))) ==NULL){
        fprintf(stderr, "Error allocating the edge image.\n");
        exit(1);
    }
    apply_hysteresis(magnitude, nms, rows, cols, tlow, thigh, *edge);

    /****************************************************************************
    * Free all of the memory that we allocated except for the edge image that
    * is still being used to store out result.
    ****************************************************************************/
    free(smoothedim);
    free(delta_x);
    free(delta_y);
    free(magnitude);
    free(nms);
}

/*******************************************************************************
* Procedure: radian_direction
* Purpose: To compute a direction of the gradient image from component dx and
* dy images. Because not all derriviatives are computed in the same way, this
* code allows for dx or dy to have been calculated in different ways.
*
* FOR X:  xdirtag = -1  for  [-1 0  1]
*         xdirtag =  1  for  [ 1 0 -1]
*
* FOR Y:  ydirtag = -1  for  [-1 0  1]'
*         ydirtag =  1  for  [ 1 0 -1]'
*
* The resulting angle is in radians measured counterclockwise from the
* xdirection. The angle points "up the gradient".
*******************************************************************************/
void radian_direction(short int *delta_x, short int *delta_y, int rows,
    int cols, float **dir_radians, int xdirtag, int ydirtag)
{
    float *dirim=NULL;

    /****************************************************************************
    * Allocate an image to store the direction of the gradient.
    ****************************************************************************/
    if((dirim = (float *) calloc(rows*cols, sizeof(float))) == NULL){
        fprintf(stderr, "Error allocating the gradient direction image.\n");
        exit(1);
    }
    *dir_radians = dirim;
    float *d_dir_radians;

    cudaMalloc((void **) &d_dir_radians,rows*cols*sizeof(float));
    cudaMemcpy(d_dir_radians,dir_radians,rows*cols*sizeof(float),cudaMemcpyHostToDevice);

    radian_direction_Device<<<BLOCKS,THREADS>>>(d_delta_x,d_delta_y,rows,cols,
        d_dir_radians,xdirtag,ydirtag);

    cudaMemcpy(dir_radians,d_dir_radians,rows*cols*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(d_dir_radians);
}

__global__
void radian_direction_Device(short int *d_delta_x, short int *d_delta_y, int rows,
    int cols, float *d_dir_radians, int xdirtag, int ydirtag)
{
    int r, c;
    double dx, dy;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for(r=id;r<rows;r+=step){
        for(c=0;c<cols;c++){
            dx = (double)d_delta_x[r+c];
            dy = (double)d_delta_y[r+c];

            if(xdirtag == 1) dx = -dx;
            if(ydirtag == -1) dy = -dy;

            d_dir_radians[r+c] = (float)angle_radians(dx, dy);
        }
    }
}

/*******************************************************************************
* FUNCTION: angle_radians
* PURPOSE: This procedure computes the angle of a vector with components x and
* y. It returns this angle in radians with the answer being in the range
* 0 <= angle <2*PI.
*******************************************************************************/
__device__
double angle_radians(double x, double y)
{
    double xu, yu, ang;

    xu = fabs(x);
    yu = fabs(y);

    if((xu == 0) && (yu == 0)) return(0);

    ang = atan(yu/xu);

    if(x >= 0){
        if(y >= 0) return(ang);
        else return(2*M_PI - ang);
    }
    else{
        if(y >= 0) return(M_PI - ang);
        else return(M_PI + ang);
    }
}

/*******************************************************************************
* PROCEDURE: magnitude_x_y
* PURPOSE: Compute the magnitude of the gradient. This is the square root of
* the sum of the squared derivative values.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void magnitude_x_y(short int *delta_x, short int *delta_y, int rows, int cols,
    short int **magnitude)
{
    cudaEvent_t start,end;
    float time;
    cudaEventCreate (&start);
    cudaEventCreate (&end);
    cudaEventRecord (start);

    /****************************************************************************
    * Allocate an image to store the magnitude of the gradient.
    ****************************************************************************/
    if((*magnitude = (short *) calloc(rows*cols, sizeof(short))) == NULL){
        fprintf(stderr, "Error allocating the magnitude image.\n");
        exit(1);
    }

    // Aloja memoria en Device
    cudaMalloc((void **) &d_magnitude,rows*cols*sizeof(short int));

    // Llamada a Kernel
    magnitude_x_y_Device<<<BLOCKS,THREADS>>>(d_delta_x,d_delta_y,rows,cols,d_magnitude);

    // Copia de datos desde Device a Host
    cudaMemcpy(*magnitude,d_magnitude,rows*cols*sizeof(short int),cudaMemcpyDeviceToHost);

    cudaEventRecord (end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime (&time, start, end);
    printf("Magnitude_x_y: \t\t\t\t%.3f	segundos\n", time/1000);
}

__global__
void magnitude_x_y_Device (short int *d_delta_x, short int *d_delta_y, int rows, int cols,
    short int *d_magnitude)
{
    int r, c, sq1, sq2, pos;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for(r=id;r<rows;r+=step){
        pos=r*cols;
        for(c=0;c<cols;c++,pos++){
            sq1 = (int)d_delta_x[pos] * (int)d_delta_x[pos];
            sq2 = (int)d_delta_y[pos] * (int)d_delta_y[pos];
            d_magnitude[pos] = (short)(0.5 + sqrt((double)sq1 + (double)sq2));
        }
    }
}

/*******************************************************************************
* PROCEDURE: derrivative_x_y
* PURPOSE: Compute the first derivative of the image in both the x any y
* directions. The differential filters that are used are:
*
*                                          -1
*         dx =  -1 0 +1     and       dy =  0
*                                          +1
*
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void derrivative_x_y(short int *smoothedim, int rows, int cols,
    short int **delta_x, short int **delta_y)
{
    cudaEvent_t start, end;
    float time;
    cudaEventCreate (&start);
    cudaEventCreate (&end);
    cudaEventRecord (start);

    /****************************************************************************
    * Allocate images to store the derivatives.
    ****************************************************************************/
    if(((*delta_x) = (short *) calloc(rows*cols, sizeof(short))) == NULL){
        fprintf(stderr, "Error allocating the delta_x image.\n");
        exit(1);
    }
    if(((*delta_y) = (short *) calloc(rows*cols, sizeof(short))) == NULL){
        fprintf(stderr, "Error allocating the delta_x image.\n");
        exit(1);
    }

    /****************************************************************************
    * Compute the x-derivative. Adjust the derivative at the borders to avoid
    * losing pixels.
    ****************************************************************************/
    if(VERBOSE) printf("   Computing the X-direction derivative.\n");

    //Asignacion de memoria para variables en el DEVICE
    cudaMalloc((void **) &d_delta_y,rows*cols*sizeof(short int));
    cudaMalloc((void **) &d_delta_x,rows*cols*sizeof(short int));

    //Llamadas a Kernel
    derrivative_x<<<BLOCKS,THREADS>>>(rows,cols,d_delta_x,d_smoothedim);
    derrivative_y<<<BLOCKS,THREADS>>>(rows,cols,d_delta_y,d_smoothedim);
    // derrivative_Device<<<BLOCKS,THREADS>>>(rows,cols,d_delta_x,d_delta_y,d_smoothedim);

    cudaFree(d_smoothedim);

    cudaEventRecord (end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime (&time, start, end);
    printf("Derrivative_x_y: \t\t\t%.3f	segundos\n", time/1000);
}

__global__
void derrivative_Device(int rows, int cols, short int *d_delta_x,
    short int *d_delta_y, short int *d_smoothedim)
{
    int i,j,id,step;
    id = threadIdx.x + blockIdx.x * blockDim.x;
    step = blockDim.x * gridDim.x;

    // Se calcula la derivada en X e Y de los elementos centrales. Sin bordes
    for (i = cols*(id+1); i < (rows*cols-cols); i=i+(cols*step)) {
            for (j = 1; j < cols-1; j++) {
                    d_delta_x[i+j] = d_smoothedim[i+j+1] - d_smoothedim[i+j-1];
                    d_delta_y[i+j] = d_smoothedim[i+j+cols] - d_smoothedim[i+j-cols];
            }
    }

    __syncthreads();
    // // Se calcula la derivada en X de la primer y ultima fila, sin los extremos
    for (j = (id+1); j < cols-1; j+=step) {
            d_delta_x[j] = d_smoothedim[j+1] - d_smoothedim[j-1];
            d_delta_x[cols*(rows-1)+j] = d_smoothedim[cols*(rows-1)+j+1] - d_smoothedim[cols*(rows-1)+j-1];
    }
    // // Se calcula la derivada en Y de la primer y ultima columna, sin los extremos
    for (i = (id+1)*cols; i < (cols*(rows-1)); i+=(step*cols)) {
            d_delta_y[i] = d_smoothedim[i+cols] - d_smoothedim[i-cols];
            d_delta_y[i+cols-1] = d_smoothedim[i+(cols-1)+cols] - d_smoothedim[i+(cols-1)-cols];
    }
    // // Devidada en X de la primer y ultima columna
    for(i=0; i<(cols*rows); i+=cols){
        d_delta_x[i] = d_smoothedim[i+1] - d_smoothedim[i];
        d_delta_x[i+cols-1] = d_smoothedim[i+cols-1] - d_smoothedim[i+cols-2];
    }
    // // Devidada en Y de la primer y ultima fila
    for(j=id;j<cols;j+=step){
        d_delta_y[j] = d_smoothedim[j+cols] - d_smoothedim[j];
        d_delta_y[j+(cols*rows)-cols] = d_smoothedim[j+(cols*rows)-cols] - d_smoothedim[j+(cols*rows)-cols-cols];
    }

}

__global__
void derrivative_x(int rows, int cols, short int *d_delta_x, short int *d_smoothedim)
{
    int r,pos,c,id,step;
    id = threadIdx.x + blockIdx.x * blockDim.x;
    step = blockDim.x * gridDim.x;

    for(r=id;r<rows;r+=step){
        pos = r * cols;
        d_delta_x[pos] = d_smoothedim[pos+1] - d_smoothedim[pos];
        pos++;
        for(c=1;c<(cols-1);c++,pos++){
            d_delta_x[pos] = d_smoothedim[pos+1] - d_smoothedim[pos-1];
        }
        d_delta_x[pos] = d_smoothedim[pos] - d_smoothedim[pos-1];
    }
}

__global__
void derrivative_y(int rows, int cols, short int *d_delta_y, short int *d_smoothedim)
{
    int c,r,pos,id,step;
    id = threadIdx.x + blockIdx.x * blockDim.x;
    step = blockDim.x * gridDim.x;

    for(c=id;c<cols;c+=step){
        pos = c;
        d_delta_y[pos] = d_smoothedim[pos+cols] - d_smoothedim[pos];
        pos += cols;
        for(r=1;r<(rows-1);r++,pos+=cols){
            d_delta_y[pos] = d_smoothedim[pos+cols] - d_smoothedim[pos-cols];
        }
        d_delta_y[pos] = d_smoothedim[pos] - d_smoothedim[pos-cols];
    }
}

/*******************************************************************************
* PROCEDURE: gaussian_smooth
* PURPOSE: Blur an image with a gaussian filter.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void gaussian_smooth(unsigned char *image, int rows, int cols, float sigma,
    short int **smoothedim)
{
    int  windowsize,        /* Dimension of the gaussian kernel. */
    center;            /* Half of the windowsize. */
    float *tempim,        /* Buffer for separable filter gaussian smoothing. */
    *kernel;        /* A one dimensional gaussian kernel. */
    cudaEvent_t start, end;
    float time;
    cudaEventCreate (&start);
    cudaEventCreate (&end);
    cudaEventRecord(start);

    /****************************************************************************
    * Create a 1-dimensional gaussian smoothing kernel.
    ****************************************************************************/
    if(VERBOSE) printf("   Computing the gaussian smoothing kernel.\n");
    make_gaussian_kernel(sigma, &kernel, &windowsize);
    center = windowsize / 2;

    /****************************************************************************
    * Allocate a temporary buffer image and the smoothed image.
    ****************************************************************************/
    if((tempim = (float *) calloc(rows*cols, sizeof(float))) == NULL){
        fprintf(stderr, "Error allocating the buffer image.\n");
        exit(1);
    }
    if(((*smoothedim) = (short int *) calloc(rows*cols,
        sizeof(short int))) == NULL){
            fprintf(stderr, "Error allocating the smoothed image.\n");
            exit(1);
    }

    if(VERBOSE) printf("   Bluring the image in the X-direction.\n");

    //Declaracion de variables
    unsigned char *d_image;
    float *d_kernel,*d_tempim;
    short int *d_smoothedim;

    //Asignacion de memoria para variables en el DEVICE
    cudaMalloc((void **) &d_image,rows*cols*sizeof(unsigned char));
    cudaMalloc((void **) &d_kernel,windowsize*sizeof(float));
    cudaMalloc((void **) &d_tempim,rows*cols*sizeof(float));
    cudaMalloc((void **) &d_smoothedim,rows*cols*sizeof(short int));

    //Copia de datos del HOST al DEVICE
    cudaMemcpy(d_image,image,rows*cols*sizeof(unsigned char),cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel,kernel,windowsize*sizeof(float),cudaMemcpyHostToDevice);

    //Llamada a funcion del DEVICE
    gaussian_smooth_Device<<<BLOCKS,THREADS>>>(rows,cols,center,d_image,d_kernel,d_tempim,d_smoothedim);

    //Liberacion de memoria en el DEVICE
    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_tempim);

    //Liberacion de memoria del HOST
    free(tempim);
    free(kernel);

    cudaEventRecord (end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime (&time, start, end);
    printf("Gaussian_Smooth: \t\t\t%.3f	segundos\n", time/1000);
}

/*
La paralelizacion se realiza por filas. Es decir, a cada hilo le correpsonde
una cierta cantidad de filas, segun si ID. Las filas no son consecutivas.
    Ejemplo:
    Al hilo 0 le corresponde la fila 0, luego la fila 0+THREADS, y asi
    hasta que cumpla la condicion de corte.
*/
__global__
void gaussian_smooth_Device(int rows, int cols, int center, unsigned char *d_image,
    float *d_kernel, float *d_tempim, short int *d_smoothedim)
{
    int id,r,rr,c,cc,step;
    double dot;
    float sum;
    id = threadIdx.x + blockIdx.x * blockDim.x;
    step = blockDim.x * gridDim.x;

    /****************************************************************************
    * Blur in the x - direction.
    ****************************************************************************/
    for(r=0;r<rows;r++){
        for(c=id;c<cols;c+=step){
            dot = 0.0;
            sum = 0.0;
            for(cc=(-center);cc<=center;cc++){
                if(((c+cc) >= 0) && ((c+cc) < cols)){
                    dot = dot + (d_kernel[center+cc] * d_image[r*cols+(c+cc)]);
                    sum = sum + (d_kernel[center+cc]);
                }
            }
            d_tempim[r*cols+c] = dot/sum;
        }
    }

    /* Espera a que todos los hilos de un bloque terminen de realizar el
    blureado en X, para luego continuar con el blureado en Y.
    Esta espera se debe a que se necesita tener a d_tempim terminado para
    poder ser usado para calcular d_smoothedim */
    // __syncthreads();
    /****************************************************************************
    * Blur in the y - direction.
    ****************************************************************************/
    // for(r=0;r<rows;r++){
    //     for(c=id;c<cols;c+=step){
    //            sum = 0.0;
    //            dot = 0.0;
    //         for(rr=(-center);rr<=center;rr++){
    //             if(((r+rr) >= 0) && ((r+rr) < rows)){
    //                 dot += (double)d_kernel[center+rr] * (double)d_tempim[(r+rr)*cols+c];
    //                 sum = sum + d_kernel[center+rr];
    //             }
    //         }
    //         d_smoothedim[r*cols+c] = (short int)(dot*BOOSTBLURFACTOR/sum + 0.5);
    //     }
    // }
    for(r=id;r<rows;r+=step){
       for(c=0;c<cols;c++){
                sum = 0.0;
                dot = 0.0;
             for(rr=(-center);rr<=center;rr++){
                 if(((r+rr) >= 0) && ((r+rr) < rows)){
                     dot += (double)d_kernel[center+rr] * (double)d_tempim[(r+rr)*cols+c];
                     sum = sum + d_kernel[center+rr];
                 }
             }
             (d_smoothedim)[r*cols+c] = (short int)(dot*BOOSTBLURFACTOR/sum + 0.5);
         }
     }

}
/*******************************************************************************
* PROCEDURE: make_gaussian_kernel
* PURPOSE: Create a one dimensional gaussian kernel.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void make_gaussian_kernel(float sigma, float **kernel, int *windowsize)
{
    int i, center;
    float x, fx, sum=0.0;

    *windowsize = 1 + 2 * ceil(2.5 * sigma);
    center = (*windowsize) / 2;

    if(VERBOSE) printf("      The kernel has %d elements.\n", *windowsize);
    if((*kernel = (float *) calloc((*windowsize), sizeof(float))) == NULL){
        fprintf(stderr, "Error callocing the gaussian kernel array.\n");
        exit(1);
    }

    for(i=0;i<(*windowsize);i++){
        x = (float)(i - center);
        fx = pow(2.71828, -0.5*x*x/(sigma*sigma)) / (sigma * sqrt(6.2831853));
        (*kernel)[i] = fx;
        sum += fx;
    }

    for(i=0;i<(*windowsize);i++) (*kernel)[i] /= sum;

    if(VERBOSE){
        printf("The filter coefficients are:\n");
        for(i=0;i<(*windowsize);i++)
        printf("kernel[%d] = %f\n", i, (*kernel)[i]);
    }
}
//<------------------------- end canny_edge.c ------------------------->

//<------------------------- begin hysteresis.c ------------------------->
/*******************************************************************************
* FILE: hysteresis.c
* This code was re-written by Mike Heath from original code obtained indirectly
* from Michigan State University. heath@csee.usf.edu (Re-written in 1996).
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>

// #define VERBOSE 0

#define NOEDGE 255
#define POSSIBLE_EDGE 128
#define EDGE 0

/*******************************************************************************
* PROCEDURE: follow_edges
* PURPOSE: This procedure edges is a recursive routine that traces edgs along
* all paths whose magnitude values remain above some specifyable lower
* threshhold.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void follow_edges(unsigned char *edgemapptr, short *edgemagptr, short lowval,
    int cols)
{
    short *tempmagptr;
    unsigned char *tempmapptr;
    int i;
    // float thethresh;
    int x[8] = {1,1,0,-1,-1,-1,0,1},
    y[8] = {0,1,1,1,0,-1,-1,-1};

    for(i=0;i<8;i++){
        tempmapptr = edgemapptr - y[i]*cols + x[i];
        tempmagptr = edgemagptr - y[i]*cols + x[i];

        if((*tempmapptr == POSSIBLE_EDGE) && (*tempmagptr > lowval)){
            *tempmapptr = (unsigned char) EDGE;
            follow_edges(tempmapptr,tempmagptr, lowval, cols);
        }
    }
}

/*******************************************************************************
* PROCEDURE: apply_hysteresis
* PURPOSE: This routine finds edges that are above some high threshhold or
* are connected to a high pixel by a path of pixels greater than a low
* threshold.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void apply_hysteresis(short int *mag, unsigned char *nms, int rows, int cols,
    float tlow, float thigh, unsigned char *edge)
{
    int r, c, pos, numedges, highcount, lowthreshold, highthreshold,
    hist[32768];
    short int maximum_mag;// sumpix;

    cudaEvent_t start, end;
    cudaEventCreate (&start);
    cudaEventCreate (&end);
    cudaEventRecord(start);
    float time;
    /****************************************************************************
    * Initialize the edge map to possible edges everywhere the non-maximal
    * suppression suggested there could be an edge except for the border. At
    * the border we say there can not be an edge because it makes the
    * follow_edges algorithm more efficient to not worry about tracking an
    * edge off the side of the image.
    ****************************************************************************/
    for(r=0;r<32768;r++) hist[r] = 0;

    //Declaracion de variables
    unsigned char *d_edge;
    int *d_hist;

    //Asignacion de memoria para variables en el DEVICE
    cudaMalloc((void **) &d_edge,rows*cols*sizeof(unsigned char));
    cudaMalloc((void **) &d_hist,32768*sizeof(int));

    //Copia de datos del HOST al DEVICE
    cudaMemcpy(d_hist, hist, 32768*sizeof(int), cudaMemcpyHostToDevice);

    //Llamada a funcion del DEVICE
    apply_hysteresis_edge <<< BLOCKS, THREADS >>> (rows, cols, d_result, d_edge,d_hist,d_magnitude);

    //Copia de datos del DEVICE al HOST
    cudaMemcpy(edge, d_edge, rows*cols*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(hist, d_hist, 32768*sizeof(int), cudaMemcpyDeviceToHost);

    //Liberacion de memoria en el DEVICE
    cudaFree(d_magnitude);
    cudaFree(d_hist);
    cudaFree(d_result);

    /****************************************************************************
    * Compute the number of pixels that passed the nonmaximal suppression.
    ****************************************************************************/
    for(r=1,numedges=0;r<32768;r++){
        if(hist[r] != 0) maximum_mag = r;
        numedges += hist[r];
    }

    highcount = (int)(numedges * thigh + 0.5);

    /****************************************************************************
    * Compute the high threshold value as the (100 * thigh) percentage point
    * in the magnitude of the gradient histogram of all the pixels that passes
    * non-maximal suppression. Then calculate the low threshold as a fraction
    * of the computed high threshold value. John Canny said in his paper
    * "A Computational Approach to Edge Detection" that "The ratio of the
    * high to low threshold in the implementation is in the range two or three
    * to one." That means that in terms of this implementation, we should
    * choose tlow ~= 0.5 or 0.33333.
    ****************************************************************************/
    r = 1;
    numedges = hist[1];
    while((r<(maximum_mag-1)) && (numedges < highcount)){
        r++;
        numedges += hist[r];
    }
    highthreshold = r;
    lowthreshold = (int)(highthreshold * tlow + 0.5);

    if(VERBOSE){
        printf("The input low and high fractions of %f and %f computed to\n",
        tlow, thigh);
        printf("magnitude of the gradient threshold values of: %d %d\n",
        lowthreshold, highthreshold);
    }

    /****************************************************************************
    * This loop looks for pixels above the highthreshold to locate edges and
    * then calls follow_edges to continue the edge.
    ****************************************************************************/
    for(r=0,pos=0;r<rows;r++){
        for(c=0;c<cols;c++,pos++){
            if((edge[pos] == POSSIBLE_EDGE) && (mag[pos] >= highthreshold)){
                edge[pos] = EDGE;
                follow_edges((edge+pos), (mag+pos), lowthreshold, cols);
            }
        }
    }


    /****************************************************************************
    * Set all the remaining possible edges to non-edges.
    ****************************************************************************/
    // Setear pixeles que no son bordes como NOEDGE
    for(r=0; r<rows; r++){
        for(c=0; c<cols; c++)
        if(edge[r*cols+c] != EDGE)
        edge[r*cols+c] = NOEDGE;
    }
    cudaFree(d_edge);

    cudaEventRecord (end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime (&time, start, end);
    printf("Apply_Hysteresis: \t\t\t%.3f	segundos\n", time/1000);
}

__global__
void apply_hysteresis_edge(int rows, int cols, unsigned char *nms,
    unsigned char *edge, int *hist,  short int *mag)
{
    int id,r,c,step;

    id = threadIdx.x + blockIdx.x * blockDim.x;
    step = blockDim.x * gridDim.x;

    // Crear matriz edge con Posibles Bordes
    for(r=id; r<rows; r+=step){
        for(c=0; c<cols; c++){
            if(nms[r*cols+c] == POSSIBLE_EDGE)
            edge[r*cols+c] = POSSIBLE_EDGE;
            else
            edge[r*cols+c] = NOEDGE;
        }
    }

    // Formatear Primer y Ultima Columna
    for(r=id; r<rows; r+=step){
        edge[r*cols] = NOEDGE;
        edge[r*cols+cols-1] = NOEDGE;
    }
    // Formatear Primer y Ultima Fila
    for(c=id; c<cols; c+=step){
        edge[c] = NOEDGE;
        edge[rows*cols-cols+c] = NOEDGE;
    }

    __syncthreads();
    /****************************************************************************
    * Compute the histogram of the magnitude image. Then use the histogram to
    * compute hysteresis thresholds.
    ****************************************************************************/
    // Crear histograma de margnitudes
    for(r=id; r<rows; r+=step){
        for(c=0; c<cols; c++){
            if(edge[r*cols+c] == POSSIBLE_EDGE)
            atomicAdd(&hist[mag[r*cols+c]], 1);
        }
    }
}

/*******************************************************************************
* PROCEDURE: non_max_supp
* PURPOSE: This routine applies non-maximal suppression to the magnitude of
* the gradient image.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void non_max_supp(short *mag, short *gradx, short *grady, int nrows, int ncols,
    unsigned char *result)
{
    cudaEvent_t start, end;
    float time;
    cudaEventCreate (&start);
    cudaEventCreate (&end);
    cudaEventRecord (start);
    int count;
    unsigned char *resultrowptr, *resultptr;

    /****************************************************************************
    * Zero the edges of the result image.
    ****************************************************************************/
    for(count=0,resultrowptr=result,resultptr=result+ncols*(nrows-1);
    count<ncols; resultptr++,resultrowptr++,count++){
        *resultrowptr = *resultptr = (unsigned char) 0;
    }
    for(count=0,resultptr=result,resultrowptr=result+ncols-1;
        count<nrows; count++,resultptr+=ncols,resultrowptr+=ncols){
            *resultptr = *resultrowptr = (unsigned char) 0;
    }
    /****************************************************************************
    * Suppress non-maximum points.
    ****************************************************************************/
    //Asignacion de memoria para variables en el DEVICE
    cudaMalloc ((void **) &d_result, nrows*ncols*sizeof(unsigned char));

    //Copia de datos de HOST a DEVICE
    cudaMemcpy (d_result, result, nrows*ncols*sizeof(unsigned char), cudaMemcpyHostToDevice);

    //Llamada a kernel de DEVICE
    non_max_supp_device<<<BLOCKS,THREADS>>>(nrows,ncols,d_magnitude,d_delta_x,d_delta_y,d_result);

    cudaFree(d_delta_x);
    cudaFree(d_delta_y);

    cudaEventRecord (end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime (&time, start, end);
    printf("Non_max_supp: \t\t\t\t%.3f	segundos\n", time/1000);
}

__global__
void non_max_supp_device(int nrows, int ncols, short *d_mag, short *d_gradx,
    short *d_grady, unsigned char *d_result)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int step = blockDim.x * gridDim.x;

    int rowcount, colcount;
    short *magrowptr,*magptr;
    short *gxrowptr,*gxptr;
    short *gyrowptr,*gyptr;
    short z1,z2;
    short m00,gx,gy;
    double mag1,mag2,xperp,yperp;
    unsigned char *resultrowptr, *resultptr;

    magrowptr = d_mag + (id+1)*ncols + 1;
    gxrowptr = d_gradx + (id+1)*ncols + 1;
    gyrowptr = d_grady + (id+1)*ncols + 1;
    resultrowptr = d_result + (id+1)*ncols + 1;


    for(rowcount=id+1;
        rowcount<nrows-2;
        rowcount+=step,
        magrowptr+=(step*ncols),
        gyrowptr+=(step*ncols),
        gxrowptr+=(step*ncols),
        resultrowptr+=(step*ncols)){
            for(colcount=1,magptr=magrowptr,gxptr=gxrowptr,gyptr=gyrowptr,
                resultptr=resultrowptr;
                colcount<ncols-2;
                colcount++,magptr++,gxptr++,gyptr++,resultptr++){
                    m00 = *magptr;
                    if(m00 == 0){
                        *resultptr = (unsigned char) NOEDGE;
                    }
                    else{
                        xperp = -(gx = *gxptr)/((float)m00);
                        yperp = (gy = *gyptr)/((float)m00);
                    }
                    if(gx >= 0){
                        if(gy >= 0){
                            if (gx >= gy)
                            {
                                z1 = *(magptr - 1);
                                z2 = *(magptr - ncols - 1);

                                mag1 = (m00 - z1)*xperp + (z2 - z1)*yperp;

                                z1 = *(magptr + 1);
                                z2 = *(magptr + ncols + 1);

                                mag2 = (m00 - z1)*xperp + (z2 - z1)*yperp;
                            }
                            else
                            {
                                z1 = *(magptr - ncols);
                                z2 = *(magptr - ncols - 1);

                                mag1 = (z1 - z2)*xperp + (z1 - m00)*yperp;

                                z1 = *(magptr + ncols);
                                z2 = *(magptr + ncols + 1);

                                mag2 = (z1 - z2)*xperp + (z1 - m00)*yperp;
                            }
                        }
                        else
                        {
                            if (gx >= -gy)
                            {
                                z1 = *(magptr - 1);
                                z2 = *(magptr + ncols - 1);

                                mag1 = (m00 - z1)*xperp + (z1 - z2)*yperp;

                                z1 = *(magptr + 1);
                                z2 = *(magptr - ncols + 1);

                                mag2 = (m00 - z1)*xperp + (z1 - z2)*yperp;
                            }
                            else
                            {
                                z1 = *(magptr + ncols);
                                z2 = *(magptr + ncols - 1);

                                mag1 = (z1 - z2)*xperp + (m00 - z1)*yperp;

                                z1 = *(magptr - ncols);
                                z2 = *(magptr - ncols + 1);

                                mag2 = (z1 - z2)*xperp  + (m00 - z1)*yperp;
                            }
                        }
                    }
                    else
                    {
                        if ((gy = *gyptr) >= 0)
                        {
                            if (-gx >= gy)
                            {
                                z1 = *(magptr + 1);
                                z2 = *(magptr - ncols + 1);

                                mag1 = (z1 - m00)*xperp + (z2 - z1)*yperp;

                                z1 = *(magptr - 1);
                                z2 = *(magptr + ncols - 1);

                                mag2 = (z1 - m00)*xperp + (z2 - z1)*yperp;
                            }
                            else
                            {
                                z1 = *(magptr - ncols);
                                z2 = *(magptr - ncols + 1);

                                mag1 = (z2 - z1)*xperp + (z1 - m00)*yperp;

                                z1 = *(magptr + ncols);
                                z2 = *(magptr + ncols - 1);

                                mag2 = (z2 - z1)*xperp + (z1 - m00)*yperp;
                            }
                        }
                        else
                        {
                            if (-gx > -gy)
                            {
                                z1 = *(magptr + 1);
                                z2 = *(magptr + ncols + 1);

                                mag1 = (z1 - m00)*xperp + (z1 - z2)*yperp;

                                z1 = *(magptr - 1);
                                z2 = *(magptr - ncols - 1);

                                mag2 = (z1 - m00)*xperp + (z1 - z2)*yperp;
                            }
                            else
                            {
                                z1 = *(magptr + ncols);
                                z2 = *(magptr + ncols + 1);

                                mag1 = (z2 - z1)*xperp + (m00 - z1)*yperp;

                                z1 = *(magptr - ncols);
                                z2 = *(magptr - ncols - 1);

                                mag2 = (z2 - z1)*xperp + (m00 - z1)*yperp;
                            }
                        }
                    }

                    if ((mag1 > 0.0) || (mag2 > 0.0))
                    *resultptr = (unsigned char) NOEDGE;
                    else
                    {
                        if (mag2 == 0.0)
                        *resultptr = (unsigned char) NOEDGE;
                        else
                        *resultptr = (unsigned char) POSSIBLE_EDGE;
                    }
            }
        }
}
//<------------------------- end hysteresis.c ------------------------->

//<------------------------- begin pgm_io.c------------------------->
/*******************************************************************************
* FILE: pgm_io.c
* This code was written by Mike Heath. heath@csee.usf.edu (in 1995).
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/******************************************************************************
* Function: read_pgm_image
* Purpose: This function reads in an image in PGM format. The image can be
* read in from either a file or from standard input. The image is only read
* from standard input when infilename = NULL. Because the PGM format includes
* the number of columns and the number of rows in the image, these are read
* from the file. Memory to store the image is allocated in this function.
* All comments in the header are discarded in the process of reading the
* image. Upon failure, this function returns 0, upon sucess it returns 1.
******************************************************************************/
int read_pgm_image(char *infilename, unsigned char **image, int *rows,
    int *cols)
{
    FILE *fp;
    char buf[71];

    /***************************************************************************
    * Open the input image file for reading if a filename was given. If no
    * filename was provided, set fp to read from standard input.
    ***************************************************************************/
    if(infilename == NULL) fp = stdin;
    else{
        if((fp = fopen(infilename, "r")) == NULL){
            fprintf(stderr, "Error reading the file %s in read_pgm_image().\n",
            infilename);
            return(0);
        }
    }

    /***************************************************************************
    * Verify that the image is in PGM format, read in the number of columns
    * and rows in the image and scan past all of the header information.
    ***************************************************************************/
    fgets(buf, 70, fp);
    if(strncmp(buf,"P5",2) != 0){
        fprintf(stderr, "The file %s is not in PGM format in ", infilename);
        fprintf(stderr, "read_pgm_image().\n");
        if(fp != stdin) fclose(fp);
        return(0);
    }
    do{ fgets(buf, 70, fp); }while(buf[0] == '#');  /* skip all comment lines */
    sscanf(buf, "%d %d", cols, rows);
    do{ fgets(buf, 70, fp); }while(buf[0] == '#');  /* skip all comment lines */

    /***************************************************************************
    * Allocate memory to store the image then read the image from the file.
    ***************************************************************************/
    if(((*image) = (unsigned char *) malloc((*rows)*(*cols))) == NULL){
        fprintf(stderr, "Memory allocation failure in read_pgm_image().\n");
        if(fp != stdin) fclose(fp);
        return(0);
    }
    if((*rows) != fread((*image), (*cols), (*rows), fp)){
        fprintf(stderr, "Error reading the image data in read_pgm_image().\n");
        if(fp != stdin) fclose(fp);
        free((*image));
        return(0);
    }

    if(fp != stdin) fclose(fp);
    return(1);
}

/******************************************************************************
* Function: write_pgm_image
* Purpose: This function writes an image in PGM format. The file is either
* written to the file specified by outfilename or to standard output if
* outfilename = NULL. A comment can be written to the header if coment != NULL.
******************************************************************************/
int write_pgm_image(char *outfilename, unsigned char *image, int rows,
    int cols, char *comment, int maxval)
{
    FILE *fp;

    /***************************************************************************
    * Open the output image file for writing if a filename was given. If no
    * filename was provided, set fp to write to standard output.
    ***************************************************************************/
    if(outfilename == NULL) fp = stdout;
    else{
        if((fp = fopen(outfilename, "w")) == NULL){
            fprintf(stderr, "Error writing the file %s in write_pgm_image().\n",
            outfilename);
            return(0);
        }
    }

    /***************************************************************************
    * Write the header information to the PGM file.
    ***************************************************************************/
    fprintf(fp, "P5\n%d %d\n", cols, rows);
    if(comment != NULL)
    if(strlen(comment) <= 70) fprintf(fp, "# %s\n", comment);
    fprintf(fp, "%d\n", maxval);

    /***************************************************************************
    * Write the image data to the file.
    ***************************************************************************/
    if(rows != fwrite(image, cols, rows, fp)){
        fprintf(stderr, "Error writing the image data in write_pgm_image().\n");
        if(fp != stdout) fclose(fp);
        return(0);
    }

    if(fp != stdout) fclose(fp);
    return(1);
}

/******************************************************************************
* Function: read_ppm_image
* Purpose: This function reads in an image in PPM format. The image can be
* read in from either a file or from standard input. The image is only read
* from standard input when infilename = NULL. Because the PPM format includes
* the number of columns and the number of rows in the image, these are read
* from the file. Memory to store the image is allocated in this function.
* All comments in the header are discarded in the process of reading the
* image. Upon failure, this function returns 0, upon sucess it returns 1.
******************************************************************************/
int read_ppm_image(char *infilename, unsigned char **image_red,
    unsigned char **image_grn, unsigned char **image_blu, int *rows,
    int *cols)
{
    FILE *fp;
    char buf[71];
    int p, size;

    /***************************************************************************
    * Open the input image file for reading if a filename was given. If no
    * filename was provided, set fp to read from standard input.
    ***************************************************************************/
    if(infilename == NULL) fp = stdin;
    else{
        if((fp = fopen(infilename, "r")) == NULL){
            fprintf(stderr, "Error reading the file %s in read_ppm_image().\n",
            infilename);
            return(0);
        }
    }

    /***************************************************************************
    * Verify that the image is in PPM format, read in the number of columns
    * and rows in the image and scan past all of the header information.
    ***************************************************************************/
    fgets(buf, 70, fp);
    if(strncmp(buf,"P6",2) != 0){
        fprintf(stderr, "The file %s is not in PPM format in ", infilename);
        fprintf(stderr, "read_ppm_image().\n");
        if(fp != stdin) fclose(fp);
        return(0);
    }
    do{ fgets(buf, 70, fp); }while(buf[0] == '#');  /* skip all comment lines */
    sscanf(buf, "%d %d", cols, rows);
    do{ fgets(buf, 70, fp); }while(buf[0] == '#');  /* skip all comment lines */

    /***************************************************************************
    * Allocate memory to store the image then read the image from the file.
    ***************************************************************************/
    if(((*image_red) = (unsigned char *) malloc((*rows)*(*cols))) == NULL){
        fprintf(stderr, "Memory allocation failure in read_ppm_image().\n");
        if(fp != stdin) fclose(fp);
        return(0);
    }
    if(((*image_grn) = (unsigned char *) malloc((*rows)*(*cols))) == NULL){
        fprintf(stderr, "Memory allocation failure in read_ppm_image().\n");
        if(fp != stdin) fclose(fp);
        return(0);
    }
    if(((*image_blu) = (unsigned char *) malloc((*rows)*(*cols))) == NULL){
        fprintf(stderr, "Memory allocation failure in read_ppm_image().\n");
        if(fp != stdin) fclose(fp);
        return(0);
    }

    size = (*rows)*(*cols);
    for(p=0;p<size;p++){
        (*image_red)[p] = (unsigned char)fgetc(fp);
        (*image_grn)[p] = (unsigned char)fgetc(fp);
        (*image_blu)[p] = (unsigned char)fgetc(fp);
    }

    if(fp != stdin) fclose(fp);
    return(1);
}

/******************************************************************************
* Function: write_ppm_image
* Purpose: This function writes an image in PPM format. The file is either
* written to the file specified by outfilename or to standard output if
* outfilename = NULL. A comment can be written to the header if coment != NULL.
******************************************************************************/
int write_ppm_image(char *outfilename, unsigned char *image_red,
    unsigned char *image_grn, unsigned char *image_blu, int rows,
    int cols, char *comment, int maxval)
{
    FILE *fp;
    long size, p;

    /***************************************************************************
    * Open the output image file for writing if a filename was given. If no
    * filename was provided, set fp to write to standard output.
    ***************************************************************************/
    if(outfilename == NULL) fp = stdout;
    else{
        if((fp = fopen(outfilename, "w")) == NULL){
            fprintf(stderr, "Error writing the file %s in write_pgm_image().\n",
            outfilename);
            return(0);
        }
    }

    /***************************************************************************
    * Write the header information to the PGM file.
    ***************************************************************************/
    fprintf(fp, "P6\n%d %d\n", cols, rows);
    if(comment != NULL)
    if(strlen(comment) <= 70) fprintf(fp, "# %s\n", comment);
    fprintf(fp, "%d\n", maxval);

    /***************************************************************************
    * Write the image data to the file.
    ***************************************************************************/
    size = (long)rows * (long)cols;
    for(p=0;p<size;p++){      /* Write the image in pixel interleaved format. */
        fputc(image_red[p], fp);
        fputc(image_grn[p], fp);
        fputc(image_blu[p], fp);
    }

    if(fp != stdout) fclose(fp);
    return(1);
}
//<------------------------- end pgm_io.c ------------------------->
