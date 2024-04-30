
#include "main.h"

using namespace std;

int main(int argc, char* argv[])
{
#if __AVX2__
    DEBUG_PRINT("AVX2 available");
#endif
#if __AVX512BW__
    DEBUG_PRINT("AVX512BW available");
#endif
#if __AVX512F__
    DEBUG_PRINT("AVX512F available");
#endif
#if __AVX512VL__
    DEBUG_PRINT("AVX512VL available");
#endif
#if __AVX512VPOPCNTDQ__
    DEBUG_PRINT("AVX512VPOPCNTDQ available");
#endif

#if !_OPENMP
    #warning "Compiling without OpenMP support."
#endif

    Arguments args = parseArguments(argc, argv); // parse command line arguments
    omp_set_num_threads(args.threads);           // set number of used threads 
    
    if (args.compress)
    {
        Compressor compressor(args.model, args.adaptive, args.width, args.threads);
        compressor.compress(args.inputFileName, args.outputFileName);
    }
    else
    {
        Decompressor decompressor(args.threads);
        decompressor.decompress(args.inputFileName, args.outputFileName);
    }
    
    return 0;
}

Arguments parseArguments(int argc, char* argv[])
{
    constexpr uint16_t DEFAULT_NUMBER_OF_THREADS = 4;

    Arguments args;
    args.compress = false;
    args.decompress = false;
    args.model = false;
    args.adaptive = false;
    args.width = 0;
    args.inputFileName = "";
    args.outputFileName = "";
#if _OPENMP
    args.threads = DEFAULT_NUMBER_OF_THREADS;
#else
    args.threads = 1; // no OpenMP support, only one thread
#endif

    bool widthSet = false;

    vector<string> arguments(argv, argv + argc);
    for (size_t i = 1; i < arguments.size(); i++) // parse arguments one by one, skipping the program name
    {
        if (arguments[i] == "-c")
        {
            args.compress = true;
        }
        else if (arguments[i] == "-d")
        {
            args.decompress = true;
        }
        else if (arguments[i] == "-m")
        {
            args.model = true;
        }
        else if (arguments[i] == "-a")
        {
            args.adaptive = true;
        }
        else if (arguments[i] == "-w")
        {
            try
            {
                args.width = stoul(arguments[++i]);
            }
            catch (const invalid_argument& e)
            {
                cerr << "Error: Unsigned integer expected after the '-w' switch, got '" << arguments[i] << "'." << endl;
                exit(1);
            }
            catch (const out_of_range& e)
            {
                cerr << "Error: Specified width is out of range with '" << arguments[i] << "'." << endl;
                exit(1);
            }

            if (args.width >= (1UL << HuffmanRLECompression::MAX_BITS_PER_FILE_DIMENSION))
            {
                cerr << "Error: Specified width is too large, maximum allowed width is " 
                     << (1UL << HuffmanRLECompression::MAX_BITS_PER_FILE_DIMENSION) - 1 << " (2^" << HuffmanRLECompression::MAX_BITS_PER_FILE_DIMENSION << "-1)." << endl;
                exit(1);
            }

            widthSet = true;
        }
        else if (arguments[i] == "-i")
        {
            i++;
            if (i < arguments.size())
            {
                args.inputFileName = arguments[i];
            }
            else
            {
                cerr << "Error: Missing input file name after the switch '-i'." << endl;
                exit(1);
            }
        }
        else if (arguments[i] == "-o")
        {
            i++;
            if (i < arguments.size())
            {
                args.outputFileName = arguments[i];
            }
            else
            {
                cerr << "Error: Missing output file name after the switch '-o'." << endl;
                exit(1);
            }
        }
        else if (arguments[i] == "-h")
        {
            cout << "Usage: " << arguments[0] << " [-c | -d] [-m] [-a] [-w <width>] [-i <input_file>] [-o <output_file>]" << endl;
            cout << "Options:" << endl;
            cout << "  -c:               Compress the input file." << endl;
            cout << "  -d:               Decompress the input file." << endl;
            cout << "  -m:               Use the model-based compression." << endl;
            cout << "  -a:               Use the adaptive model-based compression." << endl;
            cout << "  -w <width>:       Width of the compressed image." << endl;
            cout << "  -i <input_file>:  Specify the input file." << endl;
            cout << "  -o <output_file>: Specify the output file." << endl;
            cout << "  -t <threads>:     Number of threads to use (default is " << DEFAULT_NUMBER_OF_THREADS << "), must be a power of 2." << endl;
            exit(0);
        }
        else if (arguments[i] == "-t")
        {
        #if _OPENMP
            try
            {
                uint64_t threads = stoul(arguments[++i]);
                if (threads > 0 && threads <= 32)
                {
                    args.threads = threads;
                    if (popcount(threads) != 1)
                    {
                        uint64_t leadingZeros = 63 - countl_zero(threads);
                        args.threads = 1 << leadingZeros; // round to the nearest smaller power of 2
                        cerr << "Warning: Number of threads must be a power of 2, adjusted to " << args.threads << "." << endl;
                    }
                }
                else
                {
                    cerr << "Warning: Number of threads must be a power of 2 between 1 and 32 inclusive, got '" << arguments[i] << "'" << endl;
                    cerr << "         Number of threads is set to the default value of " << DEFAULT_NUMBER_OF_THREADS << "." << endl;
                }
            }
            catch (const invalid_argument& e)
            {
                cerr << "Warning: Unsigned integer expected after the '-t' switch, got '" << arguments[i] << "'." << endl;
                cerr << "         Number of threads is set to the default value of " << DEFAULT_NUMBER_OF_THREADS << "." << endl;
            }
            catch (const out_of_range& e)
            {
                cerr << "Warning: Specified number of threads is out of range with '" << arguments[i] << "'." << endl;
                cerr << "         Number of threads is set to the default value of " << DEFAULT_NUMBER_OF_THREADS << "." << endl;
            }
        #else
            cerr << "Warning: OpenMP is not supported, number of threads is ignored." << endl;
            i++;
        #endif
        }
        else
        {
            cerr << "Warning: Unknown switch '" << arguments[i] << "'." << endl;
        }
    }

    // check validity of arguments
    if (args.compress && args.decompress)
    {
        cerr << "Error: Cannot compress and decompress at the same time." << endl;
        exit(1);
    }

    if (!args.compress && !args.decompress)
    {
        cerr << "Error: Either compress or decompress must be specified." << endl;
        exit(1);
    }

    if (args.inputFileName.empty())
    {
        cerr << "Error: Input file name is missing." << endl;
        exit(1);
    }

    if (args.outputFileName.empty())
    {
        cerr << "Error: Output file name is missing." << endl;
        exit(1);
    }

    if (!widthSet && args.compress)
    {
        cerr << "Error: Width ('-w' switch) is missing." << endl;
        exit(1);
    }

    if (widthSet && args.decompress)
    {
        cerr << "Warning: Width ('-w' switch) is ignored when decompressing." << endl;
    }

    return args;
}