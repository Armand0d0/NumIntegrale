#if !defined(__UTILS__)
#define __UTILS__
//#define uchar unsigned char
class Utils{

        
        public:
                
                Utils();
                unsigned char** read_mnist_images(std::string full_path, int& number_of_images, int& image_size);
                unsigned char* read_mnist_labels(std::string full_path, int& number_of_labels);
                float* ReadFileToFoats(const char* filename,int N);
                void WriteFloatsToFile(std::string filename, float* f, int N);
                
};


#endif

