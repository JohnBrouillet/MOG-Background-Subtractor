## Mixture of Gaussians - Background Subtractor
Background subtractor using mixture of gaussians for moving objects detection.
The implementation is based on Stauffer and Grimson algorithm [1].

## Requirements
You only need openCV 3. Building openCV with openMP and the MKL, and compile with -fopenmp (and -O2 and -march too) option is highly recommended to get the best FPS rate.

## Usage
Constructor parameters are the number of gaussians to model the background and the foreground for one pixel (3 to 5), the learning rate, the minimum weight for a gaussian distribution to be classified as a background and a coefficient to reduce the size of the image for better calculation time. By default, these parameters are set to 3, 1, 0.9 and 0.5. See [1] and [2] to set the learning rate and the minimum weight.

    MOGBackgroundSubtraction(int _K = 3, float _a = 0.9, float _T = 0.5, int _downsample = 1);

Just two methods are needed :

    void init(std::vector<Mat>& imgs); 
    Mat createMask(Mat& img);
  
Give to the first one a vector of cv::Mat to initialize the K gaussians of each pixels (The first N > 10 images in the sequence/video). Then, use createMask to create the mask of the next images. Color black corresponds to the background and white to the foreground. 

Images sequences can be found here : https://sites.google.com/site/backgroundsubtraction/test-sequences

## Example
main file provide a canvas to use the algorithm. You can use an images sequence, a video or a camera. To do so, you have to provide 2 parameters : "mode" "path"

    ./mog img directory_with_image/
    ./mog video video.avi   
    ./mog cam 0

If you have several cameras, change 0 by 1 or 2 or ...

In utils file, you can find a function that reads a directory and put all images names in a vector. 

## To do
* find a more suitable morphological operation
* automatic set of learning rate and minimum weight

## Bibliography
> [1] Stauffer C, Grimson W. Adaptive background mixture models for real-time tracking. Proc IEEE Conf on Comp Vision and Patt Recog (CVPR 1999) 1999; 246-252

> [2] Thierry Bouwmans, Fida El Baf, Bertrand Vachon. Background Modeling using Mixture of Gaussians for Foreground Detection - A Survey. Recent Patents on Computer Science, Bentham Science Publishers, 2008, 1 (3), pp.219-237. <hal-00338206>
