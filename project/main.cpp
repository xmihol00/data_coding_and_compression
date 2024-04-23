
#include "main.h"

using namespace std;

int main(int argc, char* argv[])
{
#if !_OPENMP
    #warning "Compiling without OpenMP support."
#endif

    Arguments args = parseArguments(argc, argv);
    omp_set_num_threads(args.threads);
    
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
    Arguments args;
    args.compress = false;
    args.decompress = false;
    args.model = false;
    args.adaptive = false;
    args.width = 0;
    args.inputFileName = "";
    args.outputFileName = "";
#if _OPENMP
    args.threads = 4;
#else
    args.threads = 1;
#endif

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
                cerr << "Error: Unsigned integral expected after the '-w' switch, got '" << arguments[i] << "'." << endl;
                exit(1);
            }
            catch (const out_of_range& e)
            {
                cerr << "Error: Specified width is out of range with '" << arguments[i] << "'." << endl;
                exit(1);
            }
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
            cerr << "Usage: " << arguments[0] << " [-c | -d] [-m] [-a] [-w <width>] [-i <input_file>] [-o <output_file>]" << endl;
            cerr << "Options:" << endl;
            cerr << "  -c:               Compress the input file." << endl;
            cerr << "  -d:               Decompress the input file." << endl;
            cerr << "  -m:               Use the model-based compression." << endl;
            cerr << "  -a:               Use the adaptive model-based compression." << endl;
            cerr << "  -w <width>:       Width of the compressed image." << endl;
            cerr << "  -i <input_file>:  Specify the input file." << endl;
            cerr << "  -o <output_file>: Specify the output file." << endl;
            cerr << "  -t <threads>:     Number of threads to use (default is 4), must be a power of 2." << endl;
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
                        args.threads = 1 << leadingZeros;
                        cerr << "Warning: Number of threads must be a power of 2, adjusted to " << args.threads << "." << endl;
                    }
                }
                else
                {
                    cerr << "Warning: Number of threads must be even number between 1 and 32, got '" << arguments[i] << "', adjusted to 16." << endl;
                    args.threads = 16;
                }
            }
            catch (const invalid_argument& e)
            {
                cerr << "Error: Unsigned integral expected after the '-t' switch, got '" << arguments[i] << "'." << endl;
                exit(1);
            }
            catch (const out_of_range& e)
            {
                cerr << "Error: Specified number of threads is out of range with '" << arguments[i] << "'." << endl;
                exit(1);
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

    if (args.width == 0 && args.compress)
    {
        cerr << "Error: Width ('-w' switch) is missing." << endl;
        exit(1);
    }

    if (args.width != 0 && args.decompress)
    {
        cerr << "Warning: Width ('-w' switch) is ignored when decompressing." << endl;
    }

    return args;
}