
#include "main.h"

using namespace std;

int main(int argc, char* argv[])
{
    cout << "Hello, World!" << endl;
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
                unsigned long width = stoul(arguments[++i]);
                if (width > UINT32_MAX)
                {
                    throw out_of_range("Error: Width is out of range.");
                }
                args.width = static_cast<uint32_t>(width);
            }
            catch (const invalid_argument& e)
            {
                cerr << "Error: Unsigned integral expected after the '-w' switch, got '" << arguments[i] << "'." << endl;
            }
            catch (const out_of_range& e)
            {
                std::cerr << e.what() << '\n';
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
                cout << "Error: Missing input file name after the switch '-i'." << endl;
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
                cout << "Error: Missing output file name after the switch '-o'." << endl;
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
            exit(0);
        }
        else
        {
            cout << "Warning: Unknown switch '" << arguments[i] << "'." << endl;
        }
    }

    // check validity of arguments

    if (args.compress && args.decompress)
    {
        cout << "Error: Cannot compress and decompress at the same time." << endl;
        exit(1);
    }

    if (!args.compress && !args.decompress)
    {
        cout << "Error: Either compress or decompress must be specified." << endl;
        exit(1);
    }

    if (args.inputFileName.empty())
    {
        cout << "Error: Input file name is missing." << endl;
        exit(1);
    }

    if (args.outputFileName.empty())
    {
        cout << "Error: Output file name is missing." << endl;
        exit(1);
    }

    if (args.width == 0 && args.compress)
    {
        cout << "Error: Width ('-w' switch) is missing." << endl;
        exit(1);
    }

    if (args.width != 0 && args.decompress)
    {
        cout << "Warning: Width ('-w' switch) is ignored when decompressing." << endl;
    }

    return args;
}