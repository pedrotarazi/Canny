#include <omp.h>
#include <mpi.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define VERBOSE 0
#define BOOSTBLURFACTOR 90.0

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
void radian_direction(short int *delta_x, short int *delta_y, int rows,
                      int cols, float **dir_radians, int xdirtag, int ydirtag);
double angle_radians(double x, double y);
void non_max_supp(short *mag, short *gradx, short *grady, int nrows, int ncols, unsigned char *result);

int rank, numtasks;
int main(int argc, char *argv[])
{
        char *infilename = NULL; /* Name of the input image */
        char *dirfilename = NULL; /* Name of the output gradient direction image */
        char outfilename[128]; /* Name of the output "edge" image */
        char composedfname[128]; /* Name of the output "direction" image */
        unsigned char *image; /* The input image */
        unsigned char *edge; /* The output edge image */
        int rows, cols;      /* The dimensions of the image. */
        float sigma,         /* Standard deviation of the gaussian kernel. */
              tlow,    /* Fraction of the high threshold in hysteresis. */
              thigh;   /* High hysteresis threshold control. The actual
                          threshold is the (100 * thigh) percentage point
                          in the histogram of the magnitude of the
                          gradient image that passes non-maximal
                          suppression. */

        double ini = MPI_Wtime(); // Inicializa la cuenta de tiempo en MPI
        MPI_Init(&argc,&argv); // Inicializa la estructura de comunicación de MPI entre los procesos
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Asigna el numero de hilo al proceso actual
        MPI_Comm_size(MPI_COMM_WORLD, &numtasks); // Asigna el numero de procesos a la variable numtasks

        if(rank == 0) {
                printf("********************************************************\n");
                printf("************************ CANNY MPI *********************\n");
                printf("********************************************************\n");
        }

        /****************************************************************************
        * Get the command line arguments.
        ****************************************************************************/
        if(argc < 5) {
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
        if(read_pgm_image(infilename, &image, &rows, &cols) == 0) {
                fprintf(stderr, "Error reading the input image, %s.\n", infilename);
                exit(1);
        }

        /****************************************************************************
        * Perform the edge detection. All of the work takes place here.
        ****************************************************************************/
        if(VERBOSE) printf("Starting Canny edge detection.\n");
        if(dirfilename != NULL) {
                sprintf(composedfname, "%s_s_%3.2f_l_%3.2f_h_%3.2f.fim", infilename,
                        sigma, tlow, thigh);
                dirfilename = composedfname;
        }
        canny(image, rows, cols, sigma, tlow, thigh, &edge, dirfilename);

        if(rank == 0) {
                /****************************************************************************
                * Write out the edge image to a file.
                ****************************************************************************/
                sprintf(outfilename, "./out/imagen-mpi.pgm");
                if(VERBOSE) printf("Writing the edge iname in the file %s.\n", outfilename);
                if(write_pgm_image(outfilename, edge, rows, cols, "", 255) == 0) {
                        fprintf(stderr, "Error writing the edge image, %s.\n", outfilename);
                        exit(1);
                }
        }
        free(image);

        MPI_Finalize(); // Corta la comunicación entre los procesos y elimina los tipos de datos creados para ello
        double fin = MPI_Wtime();

        if(rank==0) {
                //---------------------------------------------------------------------
                FILE *archivo;
                char linea[6];
                int tiempoSerial = 0;
                int tiempo = (fin-ini)*100000;
                archivo = fopen("./out/tiempoSerial","r");

                if (archivo == NULL)
                        exit(1);
                else{
                        while (feof(archivo) == 0) {
                                fgets(linea,100,archivo);
                                tiempoSerial = atoi(linea);
                        }
                }
                fclose(archivo);

                printf("\nTiempo total del programa:\t\t%.3g segundos\n", fin-ini);
                printf("speedup:\t\t\t\t%.3f",(float)tiempoSerial/(tiempo));
                //---------------------------------------------------------------------
                printf("\n========================================================\n");
        }
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
        FILE *fpdir=NULL;     /* File to write the gradient image to.     */
        unsigned char *nms;   /* Points that are local maximal magnitude. */
        short int *smoothedim, /* The image after gaussian smoothing.      */
                  *delta_x, /* The first devivative image, x-direction. */
                  *delta_y, /* The first derivative image, y-direction. */
                  *magnitude; /* The magnitude of the gadient image.      */
        int r, c, pos;
        float *dir_radians=NULL; /* Gradient direction image.                */



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

        if(rank == 0) {
                /****************************************************************************
                * This option to write out the direction of the edge gradient was added
                * to make the information available for computing an edge quality figure
                * of merit.
                ****************************************************************************/
                if(fname != NULL) {
                        /*************************************************************************
                        * Compute the direction up the gradient, in radians that are
                        * specified counteclockwise from the positive x-axis.
                        *************************************************************************/
                        radian_direction(delta_x, delta_y, rows, cols, &dir_radians, -1, -1);

                        /*************************************************************************
                        * Write the gradient direction image out to a file.
                        *************************************************************************/
                        if((fpdir = fopen(fname, "wb")) == NULL) {
                                fprintf(stderr, "Error opening the file %s for writing.\n", fname);
                                exit(1);
                        }
                        fwrite(dir_radians, sizeof(float), rows*cols, fpdir);
                        fclose(fpdir);
                        free(dir_radians);
                }
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
        if((nms = (unsigned char *) calloc(rows*cols,sizeof(unsigned char)))==NULL) {
                fprintf(stderr, "Error allocating the nms image.\n");
                exit(1);
        }
        non_max_supp(magnitude, delta_x, delta_y, rows, cols, nms);

        /****************************************************************************
        * Use hysteresis to mark the edge pixels.
        ****************************************************************************/
        if(VERBOSE) printf("Doing hysteresis thresholding.\n");
        if((*edge=(unsigned char *)calloc(rows*cols,sizeof(unsigned char))) ==NULL) {
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
        int r, c, pos;
        float *dirim=NULL;
        double dx, dy;

        /****************************************************************************
        * Allocate an image to store the direction of the gradient.
        ****************************************************************************/
        if((dirim = (float *) calloc(rows*cols, sizeof(float))) == NULL) {
                fprintf(stderr, "Error allocating the gradient direction image.\n");
                exit(1);
        }
        *dir_radians = dirim;

        for(r=0,pos=0; r<rows; r++) {
                for(c=0; c<cols; c++,pos++) {
                        dx = (double)delta_x[pos];
                        dy = (double)delta_y[pos];

                        if(xdirtag == 1) dx = -dx;
                        if(ydirtag == -1) dy = -dy;

                        dirim[pos] = (float)angle_radians(dx, dy);
                }
        }
}

/*******************************************************************************
* FUNCTION: angle_radians
* PURPOSE: This procedure computes the angle of a vector with components x and
* y. It returns this angle in radians with the answer being in the range
* 0 <= angle <2*PI.
*******************************************************************************/
double angle_radians(double x, double y)
{
        double xu, yu, ang;

        xu = fabs(x);
        yu = fabs(y);

        if((xu == 0) && (yu == 0)) return(0);

        ang = atan(yu/xu);

        if(x >= 0) {
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
        double ini_mag,fin_mag;
        ini_mag = MPI_Wtime();

        short int *magnitude_temp;
        int r, c, pos, sq1, sq2;
        int elementos, x, cantidad;
        elementos = rows*cols;
        int rows_per_proc = rows / numtasks;
        int nroElemXProceso = rows_per_proc*cols;

        /****************************************************************************
        * Allocate an image to store the magnitude of the gradient.
        ****************************************************************************/
        if((*magnitude = (short *) calloc(rows*cols, sizeof(short))) == NULL) {
                fprintf(stderr, "Error allocating the magnitude image.\n");
                exit(1);
        }
        if((magnitude_temp = (short *) calloc(rows*cols, sizeof(short))) == NULL) {
                fprintf(stderr, "Error allocating the magnitude image.\n");
                exit(1);
        }

        // Cada proceso calcula su parte de margnitude y la guarda en un arreglo temporal
        for(pos=rank*nroElemXProceso; pos<(rank+1)*nroElemXProceso; pos++) {
                sq1 = (int)delta_x[pos] * (int)delta_x[pos];
                sq2 = (int)delta_y[pos] * (int)delta_y[pos];
                (magnitude_temp)[pos] = (short)(0.5 + sqrt((double)sq1 + (double)sq2));
        }
        magnitude_temp += rank*nroElemXProceso;
        // Cada proceso comparte su parte calculada con los demás
        MPI_Allgather(magnitude_temp, nroElemXProceso, MPI_SHORT, *magnitude, nroElemXProceso, MPI_SHORT, MPI_COMM_WORLD);
        // Cada proceso calcula las filas restantes y la guarda en su arreglo privado
        for(pos=numtasks*nroElemXProceso; pos<elementos; pos++) {
                sq1 = (int)delta_x[pos] * (int)delta_x[pos];
                sq2 = (int)delta_y[pos] * (int)delta_y[pos];
                (*magnitude)[pos] = (short)(0.5 + sqrt((double)sq1 + (double)sq2));
        }

        fin_mag = MPI_Wtime();
        if(rank==0)
                printf("Magnitude_x_y: \t\t\t\t%.3g	segundos\n", fin_mag - ini_mag);
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
void derrivative_x_yVersion1(short int *smoothedim, int rows, int cols,
// void derrivative_x_y(short int *smoothedim, int rows, int cols,
                     short int **delta_x, short int **delta_y)
{
        int r, c, pos;
        double inicio, fin, inicioX, finX, inicioY, finY;
        double ini_derivate,fin_derivate;
        short int *delta_x_temp, *delta_y_temp;
        short int *smoothedim_temp, *p_ini, *p_fin;
        MPI_Status estado;
        ini_derivate = MPI_Wtime();

        int elementos, x, cantidad;
        int rows_per_proc,rows_resto,cols_per_proc,cols_resto;
        elementos = rows*cols;
        rows_per_proc = rows / numtasks;
        cols_per_proc = cols / numtasks;
        rows_resto = rows % numtasks;
        cols_resto = cols % numtasks;

        /****************************************************************************
        * Allocate temporary buffer images to store the derivatives.
        ****************************************************************************/
        if(((delta_x_temp) = (short *) calloc(rows_per_proc*cols, sizeof(short))) == NULL) {
                fprintf(stderr, "Error allocating the delta_x image.\n");
                exit(1);
        }
        if(((delta_y_temp) = (short *) calloc(rows*cols, sizeof(short))) == NULL) {
                fprintf(stderr, "Error allocating the delta_x image.\n");
                exit(1);
        }

        /****************************************************************************
        * Allocate images to store the derivatives.
        ****************************************************************************/
        if(((*delta_x) = (short *) calloc(rows*cols, sizeof(short))) == NULL) {
                fprintf(stderr, "Error allocating the delta_x image.\n");
                exit(1);
        }
        if(((*delta_y) = (short *) calloc(rows*cols, sizeof(short))) == NULL) {
                fprintf(stderr, "Error allocating the delta_x image.\n");
                exit(1);
        }

        // Derivada en X
        /****************************************************************************
        * Compute the x-derivative. Adjust the derivative at the borders to avoid
        * losing pixels.
        ****************************************************************************/
        if(VERBOSE) printf("   Computing the X-direction derivative.\n");
        // Cada proceso calcula una cierta cantidad de filas de la derivada en X
        for(r=rank*rows_per_proc; r<(rank+1)*rows_per_proc; r++) {
                pos = r * cols;
                delta_x_temp[pos-rank*rows_per_proc*cols] = smoothedim[pos+1] - smoothedim[pos];
                pos++;
                for(c=1; c<(cols-1); c++,pos++) {
                        delta_x_temp[pos-rank*rows_per_proc*cols] = smoothedim[pos+1] - smoothedim[pos-1];
                }
                delta_x_temp[pos-rank*rows_per_proc*cols] = smoothedim[pos] - smoothedim[pos-1];
        }
        // Cada proceso comparte sus filas calculadas con los demas
        MPI_Allgather(delta_x_temp, rows_per_proc*cols, MPI_SHORT, *delta_x, rows_per_proc*cols, MPI_SHORT, MPI_COMM_WORLD);
        // Todos los hilos calculan la parte restante de la derivada en X
        for(r=numtasks*rows_per_proc; r<numtasks*rows_per_proc+rows_resto; r++) {
                pos = r * cols;
                (*delta_x)[pos] = smoothedim[pos+1] - smoothedim[pos];
                pos++;
                for(c=1; c<(cols-1); c++,pos++) {
                        (*delta_x)[pos] = smoothedim[pos+1] - smoothedim[pos-1];
                }
                (*delta_x)[pos] = smoothedim[pos] - smoothedim[pos-1];
        }

        // Derivada en Y
        /****************************************************************************
        * Compute the y-derivative.
        *
        ****************************************************************************/
        // Cada proceso calcula una cierta cantidad de columnas de la derivada en Y
        for(c=rank*cols_per_proc; c<(rank+1)*cols_per_proc; c++) {
                pos = c;
                delta_y_temp[pos] = smoothedim[pos+cols] - smoothedim[pos];
                pos += cols;
                for(r=1; r<(rows-1); r++,pos+=cols) {
                        delta_y_temp[pos] = smoothedim[pos+cols] - smoothedim[pos-cols];
                }
                delta_y_temp[pos] = smoothedim[pos] - smoothedim[pos-cols];
        }
        // Cada proceso comparte sus columnas calculadas con los demas
        // se usa reduce y no gather porque se calcula por columnas en una matriz vector
        MPI_Allreduce(delta_y_temp,*delta_y,rows*cols,MPI_SHORT,MPI_SUM,MPI_COMM_WORLD);
        // Todos los procesos calculan las columnas restantes
        for(c=numtasks*cols_per_proc; c<numtasks*cols_per_proc+cols_resto; c++) {
                pos = c;
                (*delta_y)[pos] = smoothedim[pos+cols] - smoothedim[pos];
                pos += cols;
                for(r=1; r<(rows-1); r++,pos+=cols) {
                        (*delta_y)[pos] = smoothedim[pos+cols] - smoothedim[pos-cols];
                }
                (*delta_y)[pos] = smoothedim[pos] - smoothedim[pos-cols];
        }

        free(delta_x_temp);
        free(delta_y_temp);

        fin_derivate = MPI_Wtime();
        if(rank==0) {
                printf("Derivative_x_y: \t\t\t%.3g	segundos\n", fin_derivate - ini_derivate);
        }
}


// void derrivative_x_yVersion2(short int *smoothedim, int rows, int cols,
void derrivative_x_y(short int *smoothedim, int rows, int cols,
        short int **delta_x, short int **delta_y)
{
   int r, c, pos,contador;
   double ini_derivate,fin_derivate;
   double inicio, fin, inicioX, finX, inicioY, finY;
   MPI_Status estado;
   ini_derivate = MPI_Wtime();

   int elementos, x, cantidad;
   int rows_per_proc,rows_resto,cols_per_proc,cols_resto;
   elementos = rows*cols;
   rows_per_proc = rows / numtasks;
   cols_per_proc = cols / numtasks;
   rows_resto = rows % numtasks;
   cols_resto = cols % numtasks;

   int nroFilasXProceso = rows/numtasks;
   int elemXBloque = nroFilasXProceso*cols;

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
   * Allocate temporary buffer images to store the derivatives.
   ****************************************************************************/
   short int *delta_x_temp;
   short int *delta_y_temp;
   short int *delta_x_temp_inicio;
   short int *delta_y_temp_inicio;
   if(((delta_x_temp) = (short *) calloc(rows*cols, sizeof(short)) ) == NULL) {
           fprintf(stderr, "Error allocating the delta_x image.\n");
           exit(1);
   }
   if(((delta_y_temp) = (short *) calloc(rows*cols, sizeof(short))) == NULL) {
           fprintf(stderr, "Error allocating the delta_x image.\n");
           exit(1);
   }

   /****************************************************************************
   * Compute the x-derivative. Adjust the derivative at the borders to avoid
   * losing pixels.
   ****************************************************************************/
   if(VERBOSE) printf("   Computing the X-direction derivative.\n");
   pos=0;
   int i,j;

   int empieza, termina;
   empieza = rank*elemXBloque;
   termina = (rank+1)*elemXBloque;
   if(rank == 0) empieza += cols;
   if( rows % numtasks == 0 ) {
     if(rank == numtasks-1) termina -= cols;
   }

   for (i = empieza; i < termina; i=i+cols) {
           for (j = 1; j < cols-1; j++) {
                   delta_x_temp[i+j] = smoothedim[i+j+1] - smoothedim[i+j-1];
                   delta_y_temp[i+j] = smoothedim[i+j+cols] - smoothedim[i+j-cols];
           }
   }

   delta_x_temp_inicio = delta_x_temp + rank*elemXBloque;
   MPI_Allgather(delta_x_temp_inicio, elemXBloque, MPI_SHORT, *delta_x, elemXBloque, MPI_SHORT, MPI_COMM_WORLD);

   delta_y_temp_inicio = delta_y_temp + rank*elemXBloque;
   MPI_Allgather(delta_y_temp_inicio, elemXBloque, MPI_SHORT, *delta_y, elemXBloque, MPI_SHORT, MPI_COMM_WORLD);

   empieza = numtasks*elemXBloque;
   termina = rows*cols-cols;
   for (i = empieza; i < termina; i=i+cols) {
     for (j = 1; j < cols-1; j++) {
       (*delta_x)[i+j] = smoothedim[i+j+1] - smoothedim[i+j-1];
       (*delta_y)[i+j] = smoothedim[i+j+cols] - smoothedim[i+j-cols];
     }
   }


   // Se calcula la derivada en X de la primer y ultima fila, sin los extremos
   for (j = 1; j < cols-1; j++) {
           (*delta_x)[j] = smoothedim[j+1] - smoothedim[j-1];
           (*delta_x)[cols*(rows-1)+j] = smoothedim[cols*(rows-1)+j+1] - smoothedim[cols*(rows-1)+j-1];
   }
   // Se calcula la derivada en Y de la primer y ultima columna, sin los extremos
   for (i = cols; i < (cols*(rows-1)); i+=cols) {
           (*delta_y)[i] = smoothedim[i+cols] - smoothedim[i-cols];
           (*delta_y)[i+cols-1] = smoothedim[i+(cols-1)+cols] - smoothedim[i+(cols-1)-cols];
   }

   // Devidada en X de la primer y ultima columna
   for(i=0; i<(cols*rows); i+=cols){
       (*delta_x)[i] = smoothedim[i+1] - smoothedim[i];
       (*delta_x)[i+cols-1] = smoothedim[i+cols-1] - smoothedim[i+cols-2];
   }
   // Devidada en Y de la primer y ultima fila
   for(j=0;j<cols;j++){
       (*delta_y)[j] = smoothedim[j+cols] - smoothedim[j];
       (*delta_y)[j+(cols*rows)-cols] = smoothedim[j+(cols*rows)-cols] - smoothedim[j+(cols*rows)-cols-cols];
   }

  free(delta_x_temp);
  free(delta_y_temp);

  fin_derivate = MPI_Wtime();
  if(rank==0) {
          printf("Derivative_x_y: \t\t\t%.3g	segundos\n", fin_derivate - ini_derivate);
  }
}

/*******************************************************************************
* PROCEDURE: gaussian_smooth
* PURPOSE: Blur an image with a gaussian filter.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void gaussian_smooth(unsigned char *image, int rows, int cols, float sigma, short int **smoothedim)
{
        int r, c, rr, cc, /* Counter variables. */
            windowsize, /* Dimension of the gaussian kernel. */
            center; /* Half of the windowsize. */
        float *tempim, *tempim_aux, /* Buffer for separable filter gaussian smoothing. */
              *kernel, /* A one dimensional gaussian kernel. */
              sum; /* Sum of the kernel weights variable. */
        // Se establece dot como double para no perder precisión
        double dot;  /* Dot product summing variable. */
        unsigned char *recvbuf;
        clock_t start, end;
        double time_in_seconds;
        int elementos, x, cantidad;
        short int *smoothedim_aux;
        int rows_per_proc, rows_resto, cols_per_proc, cols_resto;
        int pos1 = 0, pos2 = 0;

        start = clock();

        rows_per_proc = rows / numtasks;
        cols_per_proc = cols / numtasks;
        rows_resto = rows % numtasks;
        cols_resto = cols % numtasks;

        /****************************************************************************
        * Create a 1-dimensional gaussian smoothing kernel.
        ****************************************************************************/
        if(VERBOSE) printf("   Computing the gaussian smoothing kernel.\n");
        make_gaussian_kernel(sigma, &kernel, &windowsize);
        center = windowsize / 2;

        /****************************************************************************
        * Allocate a temporary buffer image and the smoothed image.
        ****************************************************************************/
        if((tempim = (float *) calloc(rows*cols, sizeof(float))) == NULL) {
                fprintf(stderr, "Error allocating the buffer image.\n");
                exit(1);
        }
        if(((*smoothedim) = (short int *) calloc(rows*cols, sizeof(short int))) == NULL) {
                fprintf(stderr, "Error allocating the smoothed image.\n");
                exit(1);
        }
        if((tempim_aux = (float *) calloc(rows_per_proc*cols, sizeof(float))) == NULL) {
                fprintf(stderr, "Error allocating the buffer image.\n");
                exit(1);
        }
        if((smoothedim_aux = (short int *) calloc(rows*cols, sizeof(short int))) == NULL) {
                fprintf(stderr, "Error allocating the buffer image.\n");
                exit(1);
        }

        /****************************************************************************
        * Blur in the x - direction.
        ****************************************************************************/
        // Cada proceso calcula el suavizado en X en una cierta cantidad de filas
        if(VERBOSE) printf("   Bluring the image in the X-direction.\n");
        for(r=rank*rows_per_proc; r<(rank+1)*rows_per_proc; r++) {
                for(c=0; c<cols; c++) {
                        dot = 0.0;
                        sum = 0.0;
                        for(cc = (-center); cc <= center; cc++) {
                                if( ((c+cc) >= 0) && ((c+cc) < cols) ) {
                                        dot += kernel[center+cc] * image[r*cols+(c+cc)];
                                        sum += kernel[center+cc];
                                }
                        }
                        tempim_aux[pos1] = dot/sum;
                        pos1++;
                }
        }
        // Cada hilo comparte con los demás sus filas calculadas
        MPI_Allgather(tempim_aux,cols*rows_per_proc,MPI_FLOAT,tempim,cols*rows_per_proc,MPI_FLOAT,MPI_COMM_WORLD);
        // Todos los procesos calculan las filas restantes para sí mismos
        for(r=numtasks*rows_per_proc; r<numtasks*rows_per_proc+rows_resto; r++) {
                for(c=0; c<cols; c++) {
                        dot = 0.0;
                        sum = 0.0;
                        for(cc=(-center); cc<=center; cc++) {
                                if(((c+cc) >= 0) && ((c+cc) < cols)) {
                                        dot += kernel[center+cc] * image[r*cols+(c+cc)];
                                        sum += kernel[center+cc];
                                }
                        }
                        tempim[pos1*numtasks+pos2] = dot/sum;
                        pos2++;
                }
        }

        /****************************************************************************
        * Blur in the y - direction.
        ****************************************************************************/
        if(VERBOSE) { if(rank == 0) printf("   Bluring the image in the Y-direction.\n"); }
        // Cada proceso calcula el suavizado en Y en una cierta cantidad de columnas
        for(c=rank*cols_per_proc; c<(rank+1)*cols_per_proc; c++) {
                for(r=0; r<rows; r++) {
                        sum = 0.0;
                        dot = 0.0;
                        for(rr=(-center); rr<=center; rr++) {
                                if(((r+rr) >= 0) && ((r+rr) < rows)) {
                                        dot += (double)kernel[center+rr] * (double)tempim[(r+rr)*cols+c];
                                        sum += kernel[center+rr];
                                }
                        }
                        smoothedim_aux[r*cols+c] = (short int)(dot*BOOSTBLURFACTOR/sum + 0.5);
                }
        }
        // El hilo 0 calcula la parte sobrante
        if(rank == 0) {
                for(c=numtasks*cols_per_proc; c<numtasks*cols_per_proc+cols_resto; c++) {                     //Si el chunk tiene resto, se deben procesar los excedentes aquí.
                        for(r=0; r<rows; r++) {
                                sum = 0.0;
                                dot = 0.0;
                                for(rr=(-center); rr<=center; rr++) {
                                        if(((r+rr) >= 0) && ((r+rr) < rows)) {
                                                dot += (double)kernel[center+rr] * (double)tempim[(r+rr)*cols+c];
                                                sum += kernel[center+rr];
                                        }
                                }
                                smoothedim_aux[r*cols+c] = (short int)(dot*BOOSTBLURFACTOR/sum + 0.5);
                        }
                }
        }
        // Todos se comparten sus columnas calculadas
        MPI_Allreduce(smoothedim_aux,*smoothedim,rows*cols,MPI_SHORT,MPI_SUM,MPI_COMM_WORLD);
        end = clock();

        free(tempim);
        free(tempim_aux);
        free(smoothedim_aux);
        free(kernel);

        if(rank==0) {
                time_in_seconds = (double)(end-start) / (double)CLOCKS_PER_SEC;
                printf("Gaussian_Smooth: \t\t\t%.3g	segundos\n", time_in_seconds);
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
        if((*kernel = (float *) calloc((*windowsize), sizeof(float))) == NULL) {
                fprintf(stderr, "Error callocing the gaussian kernel array.\n");
                exit(1);
        }

        for(i=0; i<(*windowsize); i++) {
                x = (float)(i - center);
                fx = pow(2.71828, -0.5*x*x/(sigma*sigma)) / (sigma * sqrt(6.2831853));
                (*kernel)[i] = fx;
                sum += fx;
        }

        for(i=0; i<(*windowsize); i++) (*kernel)[i] /= sum;

        if(VERBOSE) {
                printf("The filter coefficients are:\n");
                for(i=0; i<(*windowsize); i++)
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

#define VERBOSE 0

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
        float thethresh;
        int x[8] = {1,1,0,-1,-1,-1,0,1},
            y[8] = {0,1,1,1,0,-1,-1,-1};

        for(i=0; i<8; i++) {
                tempmapptr = edgemapptr - y[i]*cols + x[i];
                tempmagptr = edgemagptr - y[i]*cols + x[i];

                if((*tempmapptr == POSSIBLE_EDGE) && (*tempmagptr > lowval)) {
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
        int r, c, pos, numedges, lowcount, highcount, lowthreshold, highthreshold,
            i, hist[32768], hist_temp[32768], rr, cc;
        short int maximum_mag, sumpix;
        /****************************************************************************
        * Initialize the edge map to possible edges everywhere the non-maximal
        * suppression suggested there could be an edge except for the border. At
        * the border we say there can not be an edge because it makes the
        * follow_edges algorithm more efficient to not worry about tracking an
        * edge off the side of the image.
        ****************************************************************************/
        double inicio, fin;
        int filasAdicionales, cantidadElementos, h, cantidadH, /* *hist_temp,*/ filasXBloque, elemXBloque, checksum;
        unsigned char *nms_temp, *edge_temp;
        short int *mag_temp;
        int nroElemB = 0;

        inicio = omp_get_wtime();
        filasAdicionales = 0;
        // Si el número de filas no es divisible en partes enteras entre el número de tareas
        if(rows % numtasks != 0)
                // Calcular el número de filas adicionales a agregar para poder repartir en partes iguales
                filasAdicionales = numtasks - (rows % numtasks);
        // La cantidad de elementos o pixeles de la imagen con las filas adicionales es:
        cantidadElementos = (rows + filasAdicionales) * cols;
        // La cantidad de filas que le corresponde a cada tarea, teniendo en cuenta las filas adicionales:
        filasXBloque = (rows + filasAdicionales) / numtasks;
        // La cantidad de elementos que le corresponde a cada tarea, teniendo en cuenta las filas adicionales:
        elemXBloque = cantidadElementos / numtasks;

        if((nms_temp = (unsigned char *) calloc(elemXBloque, sizeof(unsigned char))) == NULL) {
                printf("Error al inicializar nms_temp!!! en rank %d\n", rank );
                exit(1);
        }

        if((edge_temp = (unsigned char *) calloc(rows*cols, sizeof(unsigned char))) == NULL) {
                printf("Error al inicializar edge_temp!!! en rank %d\n", rank );
                exit(1);
        }

        inicio = omp_get_wtime();

        // Posicionamos cada proceso al principio de su porcion de filas
        int posInicialPrimerElemento = 0, posInicialPrimerElementoSig = 0;
        posInicialPrimerElemento = rank * (rows*cols)/numtasks;
        posInicialPrimerElementoSig = (rank+1) * (rows*cols)/numtasks;

        // FOR 1. MPI: for's *1 5 7 8
        // Crea una copia de nms_temp en edge_temp en donde los posibles valores son POSSIBLE_EDGE y NOEDGE
        // Paralelo
        // Cada proceso calcula todo edge
        for(pos = 0; pos < rows*cols; pos++) {
                if(nms[pos] == POSSIBLE_EDGE)
                        edge_temp[pos] = POSSIBLE_EDGE;
                else
                        edge_temp[pos] = NOEDGE;
        }

        // FOR 2
        // Setear primer y ultima COLUMNA como NOEDGE
        for(r=0,pos=0; r<rows; r++,pos+=cols) {
                edge_temp[pos] = NOEDGE;
                edge_temp[pos+cols-1] = NOEDGE;
        }
        // FOR 3
        // Setear primer y ultima FILA como NOEDGE
        pos = (rows-1) * cols;
        for(c=0; c<cols; c++,pos++) {
                edge_temp[c] = NOEDGE;
                edge_temp[pos] = NOEDGE;
        }

        /****************************************************************************
        * Compute the histogram of the magnitude image. Then use the histogram to
        * compute hysteresis thresholds.
        ****************************************************************************/
        for(r=0; r<32768; r++)
                hist[r] = 0;
        // Paralelo
        for(pos = 0; pos < rows*cols; pos++) {
                if( edge_temp[pos] == POSSIBLE_EDGE )
                        hist[mag[pos]]++;
        }

        /****************************************************************************
        * Compute the number of pixels that passed the nonmaximal suppression.
        ****************************************************************************/
        // Paralelo
        for(r=1,numedges=0; r<32768; r++) {
                if(hist[r] != 0)
                        maximum_mag = r;
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
        while((r<(maximum_mag-1)) && (numedges < highcount)) {
                r++;
                numedges += hist[r];
        }
        highthreshold = r;
        lowthreshold = (int)(highthreshold * tlow + 0.5);
        if(VERBOSE) {
                printf("The input low and high fractions of %f and %f computed to\n",
                       tlow, thigh);
                printf("magnitude of the gradient threshold values of: %d %d\n",
                       lowthreshold, highthreshold);
        }

        /****************************************************************************
        * This loop looks for pixels above the highthreshold to locate edges and
        * then calls follow_edges to continue the edge.
        ****************************************************************************/
        // Cada rank calcula su parte y busca recursivamente bordes por toda la matriz
        for(pos = posInicialPrimerElemento; pos < posInicialPrimerElementoSig; pos++)  {
                if((edge_temp[pos] == POSSIBLE_EDGE) && (mag[pos] >= highthreshold))
                {
                        edge_temp[pos] = EDGE;
                        follow_edges((edge_temp+pos), (mag+pos), lowthreshold, cols);
                }
        }

        /****************************************************************************
        * Set all the remaining possible edges to non-edges.
        ****************************************************************************/
        // Cada rank limpia su matriz
        for(pos=0; pos<rows*cols; pos++) {
                if(edge_temp[pos] != EDGE)
                        edge_temp[pos] = NOEDGE;
        }

        // Cada rank hace una operación MIN con las matrices de los demás y se transfiere al rank 0 el resultado
        MPI_Reduce (edge_temp, edge, rows*cols, MPI_UNSIGNED_CHAR, MPI_MIN, 0, MPI_COMM_WORLD);
        if(rank == 0) {

                fin = omp_get_wtime();
                printf("Apply_Hysteresis: \t\t\t%.3g	segundos\n", (fin-inicio));
        }
}
void apply_hysteresisS(short int *mag, unsigned char *nms, int rows, int cols,
	float tlow, float thigh, unsigned char *edge)
{
    clock_t ini_hyst,fin_hyst;
    ini_hyst = clock();
   int r, c, pos, numedges, lowcount, highcount, lowthreshold, highthreshold,
       i, hist[32768], rr, cc;
   short int maximum_mag, sumpix;

   /****************************************************************************
   * Initialize the edge map to possible edges everywhere the non-maximal
   * suppression suggested there could be an edge except for the border. At
   * the border we say there can not be an edge because it makes the
   * follow_edges algorithm more efficient to not worry about tracking an
   * edge off the side of the image.
   ****************************************************************************/
   for(r=0,pos=0;r<rows;r++){
      for(c=0;c<cols;c++,pos++){
	 if(nms[pos] == POSSIBLE_EDGE) edge[pos] = POSSIBLE_EDGE;
	 else edge[pos] = NOEDGE;
      }
   }

   for(r=0,pos=0;r<rows;r++,pos+=cols){
      edge[pos] = NOEDGE;
      edge[pos+cols-1] = NOEDGE;
   }
   pos = (rows-1) * cols;
   for(c=0;c<cols;c++,pos++){
      edge[c] = NOEDGE;
      edge[pos] = NOEDGE;
   }

   /****************************************************************************
   * Compute the histogram of the magnitude image. Then use the histogram to
   * compute hysteresis thresholds.
   ****************************************************************************/
   for(r=0;r<32768;r++) hist[r] = 0;

   for(r=0,pos=0;r<rows;r++){
      for(c=0;c<cols;c++,pos++){
	 if(edge[pos] == POSSIBLE_EDGE) hist[mag[pos]]++;
      }
   }
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
   for(r=0,pos=0;r<rows;r++){
      for(c=0;c<cols;c++,pos++) if(edge[pos] != EDGE) edge[pos] = NOEDGE;
   }

   fin_hyst = clock();
   double secs_hyst = (double)(fin_hyst - ini_hyst) / CLOCKS_PER_SEC;
   printf("Apply_Hysteresis: \t\t\t%.3g	segundos\n", secs_hyst);
}
void apply_hysteresis1(short int *mag, unsigned char *nms, int rows, int cols,
                      float tlow, float thigh, unsigned char *edge)
{
    int r, c, pos, numedges, lowcount, highcount, lowthreshold, highthreshold,
        i, hist[32768], hist_temp[32768], rr, cc;
    short int maximum_mag, sumpix;
    /****************************************************************************
    * Initialize the edge map to possible edges everywhere the non-maximal
    * suppression suggested there could be an edge except for the border. At
    * the border we say there can not be an edge because it makes the
    * follow_edges algorithm more efficient to not worry about tracking an
    * edge off the side of the image.
    ****************************************************************************/
    double inicio, fin;
    int filasAdicionales, cantidadElementos, h, cantidadH, /* *hist_temp,*/ filasXBloque, elemXBloque, checksum;
    unsigned char *nms_temp, *edge_temp;
    short int *mag_temp;
    int nroElemB = 0;

    inicio = omp_get_wtime();
    filasAdicionales = 0;
    // Si el número de filas no es divisible en partes enteras entre el número de tareas
    if(rows % numtasks != 0)
        // Calcular el número de filas adicionales a agregar para poder repartir en partes iguales
        filasAdicionales = numtasks - (rows % numtasks);
    // La cantidad de elementos o pixeles de la imagen con las filas adicionales es:
    cantidadElementos = (rows + filasAdicionales) * cols;
    // La cantidad de filas que le corresponde a cada tarea, teniendo en cuenta las filas adicionales:
    filasXBloque = (rows + filasAdicionales) / numtasks;
    // La cantidad de elementos que le corresponde a cada tarea, teniendo en cuenta las filas adicionales:
    elemXBloque = cantidadElementos / numtasks;

    if((nms_temp = (unsigned char *) calloc(elemXBloque, sizeof(unsigned char))) == NULL) {
      printf("Error al inicializar nms_temp!!! en rank %d\n", rank );
      exit(1);
    }

    if((edge_temp = (unsigned char *) calloc(rows*cols, sizeof(unsigned char))) == NULL) {
      printf("Error al inicializar edge_temp!!! en rank %d\n", rank );
      exit(1);
    }

    inicio = omp_get_wtime();

    int posInicialPrimerElemento = 0, posInicialPrimerElementoSig = 0;
    posInicialPrimerElemento = rank * (rows*cols)/numtasks;
    posInicialPrimerElementoSig = (rank+1) * (rows*cols)/numtasks;

    // FOR 1. MPI: for's *1 5 7 8
    // Crea una copia de nms_temp en edge_temp en donde los posibles valores son POSSIBLE_EDGE y NOEDGE
    // Paralelo
    for(pos = 0; pos < rows*cols; pos++) {
            if(nms[pos] == POSSIBLE_EDGE)
                    edge_temp[pos] = POSSIBLE_EDGE;
            else
                    edge_temp[pos] = NOEDGE;
    }

    // FOR 2
    // Setear primer y ultima COLUMNA como NOEDGE
    for(r=0,pos=0; r<rows; r++,pos+=cols) {
            edge_temp[pos] = NOEDGE;
            edge_temp[pos+cols-1] = NOEDGE;
    }
    // FOR 3
    // Setear primer y ultima FILA como NOEDGE
    pos = (rows-1) * cols;
    for(c=0; c<cols; c++,pos++) {
            edge_temp[c] = NOEDGE;
            edge_temp[pos] = NOEDGE;
    }

    /****************************************************************************
    * Compute the histogram of the magnitude image. Then use the histogram to
    * compute hysteresis thresholds.
    ****************************************************************************/
    for(r=0; r<32768; r++)
      hist[r] = 0;
    // Paralelo
    for(pos = 0; pos < rows*cols; pos++) {
                    if( edge_temp[pos] == POSSIBLE_EDGE )
                          hist[mag[pos]]++;
    }

    /****************************************************************************
    * Compute the number of pixels that passed the nonmaximal suppression.
    ****************************************************************************/
    // Paralelo
    for(r=1,numedges=0; r<32768; r++) {
      if(hist[r] != 0)
        maximum_mag = r;
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
    while((r<(maximum_mag-1)) && (numedges < highcount)) {
            r++;
            numedges += hist[r];
    }
    highthreshold = r;
    lowthreshold = (int)(highthreshold * tlow + 0.5);
    if(VERBOSE) {
            printf("The input low and high fractions of %f and %f computed to\n",
                   tlow, thigh);
            printf("magnitude of the gradient threshold values of: %d %d\n",
                   lowthreshold, highthreshold);
    }

    /****************************************************************************
    * This loop looks for pixels above the highthreshold to locate edges and
    * then calls follow_edges to continue the edge.
    ****************************************************************************/
    // Cada rank ejecuta su parte
    for(pos = posInicialPrimerElemento; pos < posInicialPrimerElementoSig; pos++)  {
      if((edge_temp[pos] == POSSIBLE_EDGE) && (mag[pos] >= highthreshold))
      {
        edge_temp[pos] = EDGE;
        follow_edges((edge_temp+pos), (mag+pos), lowthreshold, cols);
      }
    }

    /****************************************************************************
    * Set all the remaining possible edges to non-edges.
    ****************************************************************************/
    // Cada rank limpia su matriz
    for(pos=0; pos<rows*cols; pos++) {
        if(edge_temp[pos] != EDGE)
            edge_temp[pos] = NOEDGE;
    }

    // Cada rank hace una operación MIN con las matrices de los demás y se transfiere al rank 0 el resultado
    MPI_Reduce (edge_temp, edge, rows*cols, MPI_UNSIGNED_CHAR, MPI_MIN, 0, MPI_COMM_WORLD);
    if(rank == 0){

      fin = omp_get_wtime();
       printf("Apply_Hysteresis: \t\t\t%.3g	segundos\n", (fin-inicio));
    }
}


void apply_hysteresis2(short int *mag, unsigned char *nms, int rows, int cols,
                      float tlow, float thigh, unsigned char *edge)
{
        int r, c, pos, numedges, lowcount, highcount, lowthreshold, highthreshold,
            i, hist[32768], hist_temp[32768], rr, cc;
        short int maximum_mag, sumpix;
        /****************************************************************************
        * Initialize the edge map to possible edges everywhere the non-maximal
        * suppression suggested there could be an edge except for the border. At
        * the border we say there can not be an edge because it makes the
        * follow_edges algorithm more efficient to not worry about tracking an
        * edge off the side of the image.
        ****************************************************************************/
        double inicio, fin;
        int filasAdicionales, cantidadElementos, h, cantidadH, /* *hist_temp,*/ filasXBloque, elemXBloque, checksum;
        unsigned char *nms_temp, *edge_temp;
        short int *mag_temp;
        int nroElemB = 0;

        inicio = omp_get_wtime();
        filasAdicionales = 0;
        // Si el número de filas no es divisible en partes enteras entre el número de tareas
        if(rows % numtasks != 0)
                // Calcular el número de filas adicionales a agregar para poder repartir en partes iguales
                filasAdicionales = numtasks - (rows % numtasks);
        // La cantidad de elementos o pixeles de la imagen con las filas adicionales es:
        cantidadElementos = (rows + filasAdicionales) * cols;
        // La cantidad de filas que le corresponde a cada tarea, teniendo en cuenta las filas adicionales:
        filasXBloque = (rows + filasAdicionales) / numtasks;
        // La cantidad de elementos que le corresponde a cada tarea, teniendo en cuenta las filas adicionales:
        elemXBloque = cantidadElementos / numtasks;

        if((nms_temp = (unsigned char *) calloc(elemXBloque, sizeof(unsigned char))) == NULL) {
                printf("Error al inicializar nms_temp!!! en rank %d\n", rank );
                exit(1);
        }

        if((edge_temp = (unsigned char *) calloc(rows*cols, sizeof(unsigned char))) == NULL) {
                printf("Error al inicializar edge_temp!!! en rank %d\n", rank );
                exit(1);
        }

        inicio = omp_get_wtime();

        // Posicionamos cada proceso al principio de su porcion de filas
        int posInicialPrimerElemento = 0, posInicialPrimerElementoSig = 0;
        posInicialPrimerElemento = rank * (rows*cols)/numtasks;
        posInicialPrimerElementoSig = (rank+1) * (rows*cols)/numtasks;

        // FOR 1. MPI: for's *1 5 7 8
        // Crea una copia de nms_temp en edge_temp en donde los posibles valores son POSSIBLE_EDGE y NOEDGE
        // Paralelo
        // Cada proceso calcula todo edge
        for(pos = 0; pos < rows*cols; pos++) {
                if(nms[pos] == POSSIBLE_EDGE)
                        edge_temp[pos] = POSSIBLE_EDGE;
                else
                        edge_temp[pos] = NOEDGE;
        }

        // FOR 2
        // Setear primer y ultima COLUMNA como NOEDGE
        for(r=0,pos=0; r<rows; r++,pos+=cols) {
                edge_temp[pos] = NOEDGE;
                edge_temp[pos+cols-1] = NOEDGE;
        }
        // FOR 3
        // Setear primer y ultima FILA como NOEDGE
        pos = (rows-1) * cols;
        for(c=0; c<cols; c++,pos++) {
                edge_temp[c] = NOEDGE;
                edge_temp[pos] = NOEDGE;
        }

        /****************************************************************************
        * Compute the histogram of the magnitude image. Then use the histogram to
        * compute hysteresis thresholds.
        ****************************************************************************/
        for(r=0; r<32768; r++)
                hist[r] = 0;
        // Paralelo
        for(pos = 0; pos < rows*cols; pos++) {
                if( edge_temp[pos] == POSSIBLE_EDGE )
                        hist[mag[pos]]++;
        }

        /****************************************************************************
        * Compute the number of pixels that passed the nonmaximal suppression.
        ****************************************************************************/
        // Paralelo
        for(r=1,numedges=0; r<32768; r++) {
                if(hist[r] != 0)
                        maximum_mag = r;
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
        while((r<(maximum_mag-1)) && (numedges < highcount)) {
                r++;
                numedges += hist[r];
        }
        highthreshold = r;
        lowthreshold = (int)(highthreshold * tlow + 0.5);
        if(VERBOSE) {
                printf("The input low and high fractions of %f and %f computed to\n",
                       tlow, thigh);
                printf("magnitude of the gradient threshold values of: %d %d\n",
                       lowthreshold, highthreshold);
        }

        /****************************************************************************
        * This loop looks for pixels above the highthreshold to locate edges and
        * then calls follow_edges to continue the edge.
        ****************************************************************************/
        // Cada rank calcula su parte y busca recursivamente bordes por toda la matriz
        for(pos = posInicialPrimerElemento; pos < posInicialPrimerElementoSig; pos++)  {
                if((edge_temp[pos] == POSSIBLE_EDGE) && (mag[pos] >= highthreshold))
                {
                        edge_temp[pos] = EDGE;
                        follow_edges((edge_temp+pos), (mag+pos), lowthreshold, cols);
                }
        }

        /****************************************************************************
        * Set all the remaining possible edges to non-edges.
        ****************************************************************************/
        // Cada rank limpia su matriz
        for(pos=0; pos<rows*cols; pos++) {
                if(edge_temp[pos] != EDGE)
                        edge_temp[pos] = NOEDGE;
        }

        // Cada rank hace una operación MIN con las matrices de los demás y se transfiere al rank 0 el resultado
        MPI_Reduce (edge_temp, edge, rows*cols, MPI_UNSIGNED_CHAR, MPI_MIN, 0, MPI_COMM_WORLD);
        if(rank == 0) {

                fin = omp_get_wtime();
                printf("Apply_Hysteresis: \t\t\t%.3g	segundos\n", (fin-inicio));
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
        clock_t ini_non_max_supp = clock();
        int rowcount, colcount,count;
        short *magrowptr,*magptr;
        short *gxrowptr,*gxptr;
        short *gyrowptr,*gyptr,z1,z2;
        short m00,gx,gy;
        // Se declara las siguientes variables como double en vez de floating
        // para evitar la perdida de precisión
        double mag1,mag2,xperp,yperp;
        unsigned char *resultrowptr, *resultptr, *result_temp, *result_temp_posInicial;

        if ((result_temp = (unsigned char *) calloc (nrows*ncols, sizeof (unsigned char))) == NULL) {
                fprintf (stderr, "Error allocating the tempbuffer.\n");
                exit (1);
        }

        /****************************************************************************
        * Zero the edges of the result image.
        ****************************************************************************/
        result_temp_posInicial = result_temp;
        for(count=0,resultrowptr=result_temp,resultptr=result_temp+ncols*(nrows-1);
            count<ncols; resultptr++,resultrowptr++,count++) {
                *resultrowptr = *resultptr = (unsigned char) 0;
        }

        for(count=0,resultptr=result_temp,resultrowptr=result_temp+ncols-1;
            count<nrows; count++,resultptr+=ncols,resultrowptr+=ncols) {
                *resultptr = *resultrowptr = (unsigned char) 0;
        }

        /****************************************************************************
        * Suppress non-maximum points.
        ****************************************************************************/
        int empieza, termina;
        int nroFilasXProceso = nrows/numtasks;
        int nroElemXProceso = nroFilasXProceso*ncols;

        // Se definen los puntos de inicio de los punteros
        magrowptr =    mag + (rank * nroElemXProceso) + 1;
        gxrowptr =     gradx + (rank * nroElemXProceso) + 1;
        gyrowptr =     grady + (rank * nroElemXProceso) + 1;
        resultrowptr = result_temp + (rank * nroElemXProceso) + 1;

        // Se definen los puntos de inicio y fin de los contadores,
        // en correspondencia a los numeros de tarea y los anteriores punteros
        // Tarea 0
        if (rank == 0) {
                empieza =      1;
                termina =      nroFilasXProceso;

                // La tarea 0 avanza una fila a sus punteros, ya que las filas y columnas
                // exteriores no se procesan
                magrowptr += ncols;
                gxrowptr += ncols;
                gyrowptr += ncols;
                resultrowptr += ncols;
        }
        // Tareas intermedias
        else if (rank < numtasks -1) {

                empieza =      rank * nroFilasXProceso;
                termina =      (rank+1) * nroFilasXProceso;
        // Última tarea
        }else {
                empieza =      rank * nroFilasXProceso;
                // Se define la cantidad de filas restantes que deben calcular todos
                if( nrows % numtasks == 0 ) {
                        termina =      (rank+1) * nroFilasXProceso - 2;
                }else if( nrows % numtasks == 1 ) {
                        termina =      (rank+1) * nroFilasXProceso - 1;
                }else{
                        termina =      (rank+1) * nroFilasXProceso;
                }

        }

        // Cada proceso calcula su porción de filas
        for(rowcount = empieza;
            rowcount < termina;
            rowcount++,
            magrowptr+=ncols,
            gyrowptr+=ncols,
            gxrowptr+=ncols,
            resultrowptr+=ncols)
        {

                for(colcount=1,magptr=magrowptr,gxptr=gxrowptr,gyptr=gyrowptr,
                    result_temp=resultrowptr;
                    colcount<ncols-2;
                    colcount++,magptr++,gxptr++,gyptr++,result_temp++)
                {
                        m00 = *magptr;

                        if(m00 == 0) {
                                *result_temp = (unsigned char) NOEDGE;
                        }
                        else{
                                xperp = -(gx = *gxptr)/((float)m00);
                                yperp = (gy = *gyptr)/((float)m00);
                        }
                        if(gx >= 0) {
                                if(gy >= 0) {
                                        if (gx >= gy)
                                        {
                                                /* 111 */
                                                /* Left point */
                                                z1 = *(magptr - 1);
                                                z2 = *(magptr - ncols - 1);

                                                mag1 = (m00 - z1)*xperp + (z2 - z1)*yperp;

                                                /* Right point */
                                                z1 = *(magptr + 1);
                                                z2 = *(magptr + ncols + 1);

                                                mag2 = (m00 - z1)*xperp + (z2 - z1)*yperp;
                                        }
                                        else
                                        {
                                                /* 110 */
                                                /* Left point */
                                                z1 = *(magptr - ncols);
                                                z2 = *(magptr - ncols - 1);

                                                mag1 = (z1 - z2)*xperp + (z1 - m00)*yperp;

                                                /* Right point */
                                                z1 = *(magptr + ncols);
                                                z2 = *(magptr + ncols + 1);

                                                mag2 = (z1 - z2)*xperp + (z1 - m00)*yperp;
                                        }
                                }
                                else
                                {
                                        if (gx >= -gy)
                                        {
                                                /* 101 */
                                                /* Left point */
                                                z1 = *(magptr - 1);
                                                z2 = *(magptr + ncols - 1);

                                                mag1 = (m00 - z1)*xperp + (z1 - z2)*yperp;

                                                /* Right point */
                                                z1 = *(magptr + 1);
                                                z2 = *(magptr - ncols + 1);

                                                mag2 = (m00 - z1)*xperp + (z1 - z2)*yperp;
                                        }
                                        else
                                        {
                                                /* 100 */
                                                /* Left point */
                                                z1 = *(magptr + ncols);
                                                z2 = *(magptr + ncols - 1);

                                                mag1 = (z1 - z2)*xperp + (m00 - z1)*yperp;

                                                /* Right point */
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
                                                /* 011 */
                                                /* Left point */
                                                z1 = *(magptr + 1);
                                                z2 = *(magptr - ncols + 1);

                                                mag1 = (z1 - m00)*xperp + (z2 - z1)*yperp;

                                                /* Right point */
                                                z1 = *(magptr - 1);
                                                z2 = *(magptr + ncols - 1);

                                                mag2 = (z1 - m00)*xperp + (z2 - z1)*yperp;
                                        }
                                        else
                                        {
                                                /* 010 */
                                                /* Left point */
                                                z1 = *(magptr - ncols);
                                                z2 = *(magptr - ncols + 1);

                                                mag1 = (z2 - z1)*xperp + (z1 - m00)*yperp;

                                                /* Right point */
                                                z1 = *(magptr + ncols);
                                                z2 = *(magptr + ncols - 1);

                                                mag2 = (z2 - z1)*xperp + (z1 - m00)*yperp;
                                        }
                                }
                                else
                                {
                                        if (-gx > -gy)
                                        {
                                                /* 001 */
                                                /* Left point */
                                                z1 = *(magptr + 1);
                                                z2 = *(magptr + ncols + 1);

                                                mag1 = (z1 - m00)*xperp + (z1 - z2)*yperp;

                                                /* Right point */
                                                z1 = *(magptr - 1);
                                                z2 = *(magptr - ncols - 1);

                                                mag2 = (z1 - m00)*xperp + (z1 - z2)*yperp;
                                        }
                                        else
                                        {
                                                /* 000 */
                                                /* Left point */
                                                z1 = *(magptr + ncols);
                                                z2 = *(magptr + ncols + 1);

                                                mag1 = (z2 - z1)*xperp + (m00 - z1)*yperp;

                                                /* Right point */
                                                z1 = *(magptr - ncols);
                                                z2 = *(magptr - ncols - 1);

                                                mag2 = (z2 - z1)*xperp + (m00 - z1)*yperp;
                                        }
                                }
                        }

                        /* Now determine if the current point is a maximum point */

                        if ((mag1 > 0.0) || (mag2 > 0.0))
                        {
                                *result_temp = (unsigned char) NOEDGE;
                        }
                        else
                        {
                                if (mag2 == 0.0)
                                        *result_temp = (unsigned char) NOEDGE;
                                else
                                        *result_temp = (unsigned char) POSSIBLE_EDGE;
                        }
                }
        }

        // Cada proceso comparte su porcion calculada, con los demás
        result_temp = result_temp_posInicial + rank*nroElemXProceso;
        MPI_Allgather(result_temp, nroElemXProceso, MPI_UNSIGNED_CHAR,
                      result, nroElemXProceso, MPI_UNSIGNED_CHAR,
                      MPI_COMM_WORLD);


         // Todos los procesos se posicionan al principio de las filas sobrantes
        empieza = numtasks * nroFilasXProceso;
        termina = nrows-2;
        magrowptr =    mag + numtasks * nroElemXProceso + 1;
        gxrowptr =     gradx + numtasks * nroElemXProceso + 1;
        gyrowptr =     grady + numtasks * nroElemXProceso + 1;
        resultrowptr = result + numtasks * nroElemXProceso + 1;

        // Cada proceso calcula las filas sobrantes para sí mismo
        for(rowcount = empieza;
            rowcount < termina;
            rowcount++,
            magrowptr+=ncols,
            gyrowptr+=ncols,
            gxrowptr+=ncols,
            resultrowptr+=ncols)
        {
                for(colcount=1,magptr=magrowptr,gxptr=gxrowptr,gyptr=gyrowptr,
                    result=resultrowptr;
                    colcount<ncols-2;
                    colcount++,magptr++,gxptr++,gyptr++,result++)
               {
                        m00 = *magptr;
                        if(m00 == 0) {
                                *result = (unsigned char) NOEDGE;
                        }
                        else{
                                xperp = -(gx = *gxptr)/((float)m00);
                                yperp = (gy = *gyptr)/((float)m00);
                        }
                        if(gx >= 0) {
                                if(gy >= 0) {
                                        if (gx >= gy)
                                        {
                                                /* 111 */
                                                /* Left point */
                                                z1 = *(magptr - 1);
                                                z2 = *(magptr - ncols - 1);

                                                mag1 = (m00 - z1)*xperp + (z2 - z1)*yperp;

                                                /* Right point */
                                                z1 = *(magptr + 1);
                                                z2 = *(magptr + ncols + 1);

                                                mag2 = (m00 - z1)*xperp + (z2 - z1)*yperp;
                                        }
                                        else
                                        {
                                                /* 110 */
                                                /* Left point */
                                                z1 = *(magptr - ncols);
                                                z2 = *(magptr - ncols - 1);

                                                mag1 = (z1 - z2)*xperp + (z1 - m00)*yperp;

                                                /* Right point */
                                                z1 = *(magptr + ncols);
                                                z2 = *(magptr + ncols + 1);

                                                mag2 = (z1 - z2)*xperp + (z1 - m00)*yperp;
                                        }
                                }
                                else
                                {
                                        if (gx >= -gy)
                                        {
                                                /* 101 */
                                                /* Left point */
                                                z1 = *(magptr - 1);
                                                z2 = *(magptr + ncols - 1);

                                                mag1 = (m00 - z1)*xperp + (z1 - z2)*yperp;

                                                /* Right point */
                                                z1 = *(magptr + 1);
                                                z2 = *(magptr - ncols + 1);

                                                mag2 = (m00 - z1)*xperp + (z1 - z2)*yperp;
                                        }
                                        else
                                        {
                                                /* 100 */
                                                /* Left point */
                                                z1 = *(magptr + ncols);
                                                z2 = *(magptr + ncols - 1);

                                                mag1 = (z1 - z2)*xperp + (m00 - z1)*yperp;

                                                /* Right point */
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
                                                /* 011 */
                                                /* Left point */
                                                z1 = *(magptr + 1);
                                                z2 = *(magptr - ncols + 1);

                                                mag1 = (z1 - m00)*xperp + (z2 - z1)*yperp;

                                                /* Right point */
                                                z1 = *(magptr - 1);
                                                z2 = *(magptr + ncols - 1);

                                                mag2 = (z1 - m00)*xperp + (z2 - z1)*yperp;
                                        }
                                        else
                                        {
                                                /* 010 */
                                                /* Left point */
                                                z1 = *(magptr - ncols);
                                                z2 = *(magptr - ncols + 1);

                                                mag1 = (z2 - z1)*xperp + (z1 - m00)*yperp;

                                                /* Right point */
                                                z1 = *(magptr + ncols);
                                                z2 = *(magptr + ncols - 1);

                                                mag2 = (z2 - z1)*xperp + (z1 - m00)*yperp;
                                        }
                                }
                                else
                                {
                                        if (-gx > -gy)
                                        {
                                                /* 001 */
                                                /* Left point */
                                                z1 = *(magptr + 1);
                                                z2 = *(magptr + ncols + 1);

                                                mag1 = (z1 - m00)*xperp + (z1 - z2)*yperp;

                                                /* Right point */
                                                z1 = *(magptr - 1);
                                                z2 = *(magptr - ncols - 1);

                                                mag2 = (z1 - m00)*xperp + (z1 - z2)*yperp;
                                        }
                                        else
                                        {
                                                /* 000 */
                                                /* Left point */
                                                z1 = *(magptr + ncols);
                                                z2 = *(magptr + ncols + 1);

                                                mag1 = (z2 - z1)*xperp + (m00 - z1)*yperp;

                                                /* Right point */
                                                z1 = *(magptr - ncols);
                                                z2 = *(magptr - ncols - 1);

                                                mag2 = (z2 - z1)*xperp + (m00 - z1)*yperp;
                                        }
                                }
                        }

                        /* Now determine if the current point is a maximum point */

                        if ((mag1 > 0.0) || (mag2 > 0.0))
                        {
                                *result = (unsigned char) NOEDGE;
                        }
                        else
                        {
                                if (mag2 == 0.0)
                                        *result = (unsigned char) NOEDGE;
                                else
                                        *result = (unsigned char) POSSIBLE_EDGE;
                        }
                }
        }

        clock_t fin_non_max_supp = clock();
        double secs_non_max_supp = (double)(fin_non_max_supp - ini_non_max_supp) / CLOCKS_PER_SEC;
        if(rank==0) printf("Non_Max_Supp: \t\t\t\t%.3g	segundos\n", secs_non_max_supp );
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
                if((fp = fopen(infilename, "r")) == NULL) {
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
        if(strncmp(buf,"P5",2) != 0) {
                fprintf(stderr, "The file %s is not in PGM format in ", infilename);
                fprintf(stderr, "read_pgm_image().\n");
                if(fp != stdin) fclose(fp);
                return(0);
        }
        do { fgets(buf, 70, fp); } while(buf[0] == '#');                         /* skip all comment lines */
        sscanf(buf, "%d %d", cols, rows);
        do { fgets(buf, 70, fp); } while(buf[0] == '#');                         /* skip all comment lines */

        /***************************************************************************
        * Allocate memory to store the image then read the image from the file.
        ***************************************************************************/
        if(((*image) = (unsigned char *) malloc((*rows)*(*cols))) == NULL) {
                fprintf(stderr, "Memory allocation failure in read_pgm_image().\n");
                if(fp != stdin) fclose(fp);
                return(0);
        }
        if((*rows) != fread((*image), (*cols), (*rows), fp)) {
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
                if((fp = fopen(outfilename, "w")) == NULL) {
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
        if(rows != fwrite(image, cols, rows, fp)) {
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
                if((fp = fopen(infilename, "r")) == NULL) {
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
        if(strncmp(buf,"P6",2) != 0) {
                fprintf(stderr, "The file %s is not in PPM format in ", infilename);
                fprintf(stderr, "read_ppm_image().\n");
                if(fp != stdin) fclose(fp);
                return(0);
        }
        do { fgets(buf, 70, fp); } while(buf[0] == '#');                               /* skip all comment lines */
        sscanf(buf, "%d %d", cols, rows);
        do { fgets(buf, 70, fp); } while(buf[0] == '#');                               /* skip all comment lines */

        /***************************************************************************
        * Allocate memory to store the image then read the image from the file.
        ***************************************************************************/
        if(((*image_red) = (unsigned char *) malloc((*rows)*(*cols))) == NULL) {
                fprintf(stderr, "Memory allocation failure in read_ppm_image().\n");
                if(fp != stdin) fclose(fp);
                return(0);
        }
        if(((*image_grn) = (unsigned char *) malloc((*rows)*(*cols))) == NULL) {
                fprintf(stderr, "Memory allocation failure in read_ppm_image().\n");
                if(fp != stdin) fclose(fp);
                return(0);
        }
        if(((*image_blu) = (unsigned char *) malloc((*rows)*(*cols))) == NULL) {
                fprintf(stderr, "Memory allocation failure in read_ppm_image().\n");
                if(fp != stdin) fclose(fp);
                return(0);
        }

        size = (*rows)*(*cols);
        for(p=0; p<size; p++) {
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
                if((fp = fopen(outfilename, "w")) == NULL) {
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
        for(p=0; p<size; p++) {                                     /* Write the image in pixel interleaved format. */
                fputc(image_red[p], fp);
                fputc(image_grn[p], fp);
                fputc(image_blu[p], fp);
        }

        if(fp != stdout) fclose(fp);
        return(1);
}
//<------------------------- end pgm_io.c ------------------------->
