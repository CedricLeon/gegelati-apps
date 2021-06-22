#include <iostream>
#include <numeric>
#include <thread>
#include <atomic>
#include <chrono>
#include <inttypes.h>

#include <gegelati.h>

#include "mnist.h"

void getKey(std::atomic<bool>& exit)
{
	std::cout << std::endl;
	std::cout << "Press `q` then [Enter] to exit." << std::endl;
	std::cout.flush();

	exit = false;

    while (!exit)
    {
        char c;
        std::cin >> c;
        switch (c)
        {
            case 'q':
            case 'Q':
                exit = true;
                break;
            default:
                printf("Invalid key '%c' pressed.", c);
                std::cout.flush();
        }
    }

	printf("Program will terminate at the end of next generation.\n");
	std::cout.flush();
}

int main(int argc, char* argv[])
{
	std::cout << "Start MNIST application." << std::endl;

	// Create the instruction set for programs
	Instructions::Set set;
	auto minus = [](double a, double b)->double {return a - b; };
	auto add = [](double a, double b)->double {return a + b; };
	auto mult = [](double a, double b)->double {return a * b; };
	auto div = [](double a, double b)->double {return a / b; };
	auto max = [](double a, double b)->double {return std::max(a, b); };
	auto ln = [](double a)->double {return std::log(a); };
	auto exp = [](double a)->double {return std::exp(a); };
	auto sobelMagn = [](const double a[3][3])->double {
		double result = 0.0;
		double gx =
			-a[0][0] + a[0][2]
			- 2.0 * a[1][0] + 2.0 * a[1][2]
			- a[2][0] + a[2][2];
		double gy = -a[0][0] - 2.0 * a[0][1] - a[0][2]
			+ a[2][0] + 2.0 * a[2][1] + a[2][2];
		result = sqrt(gx * gx + gy * gy);
		return result;
	};

	auto sobelDir = [](const double a[3][3])->double {
		double result = 0.0;
		double gx =
			-a[0][0] + a[0][2]
			- 2.0 * a[1][0] + 2.0 * a[1][2]
			- a[2][0] + a[2][2];
		double gy = -a[0][0] - 2.0 * a[0][1] - a[0][2]
			+ a[2][0] + 2.0 * a[2][1] + a[2][2];
		result = std::atan(gy / gx);
		return result;
	};

	set.add(*(new Instructions::LambdaInstruction<double, double>(minus)));
	set.add(*(new Instructions::LambdaInstruction<double, double>(add)));
	set.add(*(new Instructions::LambdaInstruction<double, double>(mult)));
	set.add(*(new Instructions::LambdaInstruction<double, double>(div)));
	set.add(*(new Instructions::LambdaInstruction<double, double>(max)));
	set.add(*(new Instructions::LambdaInstruction<double>(exp)));
	set.add(*(new Instructions::LambdaInstruction<double>(ln)));
	set.add(*(new Instructions::LambdaInstruction<const double[3][3]>(sobelMagn)));
	set.add(*(new Instructions::LambdaInstruction<const double[3][3]>(sobelDir)));

    size_t seed = 0;
    if (argc > 1)
    {
        seed = atoi(argv[1]);
    }
    else
    {
        std::cout << "Seed was not precised, using default value: " << seed << std::endl;
    }

	// Set the parameters for the learning process.
	// (Controls mutations probability, program lengths, and graph size
	// among other things)
	// Loads them from the file params.json
	Learn::LearningParameters params;
	File::ParametersParser::loadParametersFromJson(ROOT_DIR "/params.json", params);
#ifdef NB_GENERATIONS
	params.nbGenerations = NB_GENERATIONS;
#endif // !NB_GENERATIONS


	// Instantiate the LearningEnvironment
	MNIST mnistLE(seed);

	std::cout << "Number of threads: " << params.nbThreads << std::endl;

	// Instantiate and init the learning agent
	Learn::ClassificationLearningAgent la(mnistLE, set, params);
	la.init();

	// Create an exporter for all graphs
	File::TPGGraphDotExporter dotExporter("out_0000.dot", la.getTPGGraph());

	// Start a thread for controlling the loop
/*#ifndef NO_CONSOLE_CONTROL
	std::atomic<bool> exitProgram = true; // (set to false by other thread) 
	std::atomic<bool> printStats = false;

	std::thread threadKeyboard(getKey, std::ref(exitProgram));

	while (exitProgram); // Wait for other thread to print key info.
#else 
	std::atomic<bool> exitProgram = false; // (set to false by other thread) 
	std::atomic<bool> printStats = false;
#endif*/

	// Adds a logger to the LA (to get statistics on learning) on std::cout
	Log::LABasicLogger logCout(la);

	// File for printing best policy stat.
	std::ofstream stats;
	stats.open("bestPolicyStats.md");
	Log::LAPolicyStatsLogger logStats(la, stats);

    std::string const fileClassificationTableName("/home/cleonard/dev/gegelati-apps/mnist/fileClassificationTable.txt");

	// Train for NB_GENERATIONS generations
	for (int i = 0; i < (int) params.nbGenerations /*&& !exitProgram*/; i++)
	{
        // Save best generation policy
		char buff[20];
		sprintf(buff, "out_%04d.dot", i);
		dotExporter.setNewFilePath(buff);
		dotExporter.print();

        // Train
		la.trainOneGeneration(i);

        // Print Classification Table
        mnistLE.printClassifStatsTable(la.getTPGGraph().getEnvironment(), la.getTPGGraph().getRootVertices().at(0), i, fileClassificationTableName, false);

/*		if (printStats) {
			mnistLE.printClassifStatsTable(la.getTPGGraph().getEnvironment(), la.getBestRoot().first, i, fileClassificationTableName);
			printStats = false;
		}*/
	}

	// Keep best policy
	la.keepBestPolicy();
	dotExporter.setNewFilePath("out_best.dot");
	dotExporter.print();

	TPG::PolicyStats ps;
	ps.setEnvironment(la.getTPGGraph().getEnvironment());
	ps.analyzePolicy(la.getBestRoot().first);
	std::ofstream bestStats;
	bestStats.open("out_best_stats.md");
	bestStats << ps;
	bestStats.close();

	// close log file also
	stats.close();

	// Print stats one last time
	//mnistLE.printClassifStatsTable(la.getTPGGraph().getEnvironment(), la.getTPGGraph().getRootVertices().at(0), (int) params.nbGenerations, fileClassificationTableName, false);

	// cleanup
	for (unsigned int i = 0; i < set.getNbInstructions(); i++) {
		delete (&set.getInstruction(i));
	}

/*#ifndef NO_CONSOLE_CONTROL
	// Exit the thread
	std::cout << "Exiting program, press a key then [enter] to exit if nothing happens.";
	threadKeyboard.join();
#endif*/

	return 0;
}
