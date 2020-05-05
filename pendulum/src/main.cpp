#include <iostream>
#include <numeric>
#include <thread>
#include <atomic>
#include <chrono>
#include <inttypes.h>
#define _USE_MATH_DEFINES // To get M_PI
#include <math.h>

#include "pendulum.h"
#include "render.h"

#ifndef NB_GENERATIONS
#define NB_GENERATIONS 1200
#endif

int main() {

	std::cout << "Start Pendulum application." << std::endl;

	// Create the instruction set for programs
	Instructions::Set set;
	auto minus = [](double a, double b)->double {return a - b; };
	auto add = [](double a, double b)->double {return a + b; };
	auto mult = [](double a, double b)->double {return a * b; };
	auto div = [](double a, double b)->double {return a / b; };
	auto max = [](double a, double b)->double {return std::max(a, b); };
	auto ln = [](double a, double b)->double {return std::log(a); };
	auto exp = [](double a, double b)->double {return std::exp(a); };
	auto cos = [](double a, double b)->double {return std::cos(a); };
	auto sin = [](double a, double b)->double {return std::sin(a); };
	auto tan = [](double a, double b)->double {return std::tan(a); };
	auto pi = [](double a, double b)->double {return M_PI; };

	set.add(*(new Instructions::LambdaInstruction<double>(minus)));
	set.add(*(new Instructions::LambdaInstruction<double>(add)));
	set.add(*(new Instructions::LambdaInstruction<double>(mult)));
	set.add(*(new Instructions::LambdaInstruction<double>(div)));
	set.add(*(new Instructions::LambdaInstruction<double>(max)));
	set.add(*(new Instructions::LambdaInstruction<double>(exp)));
	set.add(*(new Instructions::LambdaInstruction<double>(ln)));
	set.add(*(new Instructions::LambdaInstruction<double>(cos)));
	set.add(*(new Instructions::LambdaInstruction<double>(sin)));
	set.add(*(new Instructions::LambdaInstruction<double>(tan)));
	set.add(*(new Instructions::MultByConstParam<double, float>()));
	set.add(*(new Instructions::LambdaInstruction<double>(pi)));

	// Set the parameters for the learning process.
	// (Controls mutations probability, program lengths, and graph size
	// among other things)
	Learn::LearningParameters params;
	params.mutation.tpg.maxInitOutgoingEdges = 3;
	params.mutation.tpg.nbRoots = 2000;
	params.mutation.tpg.pEdgeDeletion = 0.7;
	params.mutation.tpg.pEdgeAddition = 0.7;
	params.mutation.tpg.pProgramMutation = 0.2;
	params.mutation.tpg.pEdgeDestinationChange = 0.1;
	params.mutation.tpg.pEdgeDestinationIsAction = 0.5;
	params.mutation.tpg.maxOutgoingEdges = 5;
	params.mutation.prog.pAdd = 0.5;
	params.mutation.prog.pDelete = 0.5;
	params.mutation.prog.pMutate = 1.0;
	params.mutation.prog.pSwap = 1.0;
	params.mutation.prog.maxProgramSize = 20;
	params.maxNbActionsPerEval = 1000;
	params.nbIterationsPerPolicyEvaluation = 5;
	params.ratioDeletedRoots = 0.998;
	params.archiveSize = 2000;
	params.archivingProbability = 0.01;

	// Instantiate the LearningEnvironment
	Pendulum pendulumLE({ 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0 });

	std::cout << "Number of threads: " << std::thread::hardware_concurrency() << std::endl;

	// Instantiate and init the learning agent
	Learn::ParallelLearningAgent la(pendulumLE, set, params);
	la.init();

	// Create an exporter for all graphs
	File::TPGGraphDotExporter dotExporter("out_0000.dot", la.getTPGGraph());


	// Start a thread for controlling the loop
#ifndef NO_CONSOLE_CONTROL
	// Console
	std::atomic<bool> exitProgram = true; // (set to false by other thread) 
	std::atomic<bool> toggleDisplay = true;
	std::atomic<bool> doDisplay = false;
	std::atomic<uint64_t> generation = 0;

	const TPG::TPGVertex* bestRoot = NULL;

	std::thread threadDisplay(Render::controllerLoop, std::ref(exitProgram), std::ref(toggleDisplay), std::ref(doDisplay),
		&bestRoot, std::ref(set), std::ref(pendulumLE), std::ref(params), std::ref(generation));

	while (exitProgram); // Wait for other thread to print key info.
#else 
	std::atomic<bool> exitProgram = false; // (set to false by other thread) 
	std::atomic<bool> toggleDisplay = false;
#endif

	// Train for NB_GENERATIONS generations
	printf("\nGen\tNbVert\tMin\tAvg\tMax\tTvalid\tTtrain\n");
	for (int i = 0; i < NB_GENERATIONS && !exitProgram; i++) {
		char buff[13];
		sprintf(buff, "out_%04d.dot", i);
		dotExporter.setNewFilePath(buff);
		dotExporter.print();
		std::multimap<std::shared_ptr<Learn::EvaluationResult>, const TPG::TPGVertex*> result;
		auto startEval = std::chrono::high_resolution_clock::now();
		result = la.evaluateAllRoots(i, Learn::LearningMode::VALIDATION);
		auto stopEval = std::chrono::high_resolution_clock::now();
		auto iter = result.begin();
		double min = iter->first->getResult();
		std::advance(iter, result.size() - 1);
		double max = iter->first->getResult();
		double avg = std::accumulate(result.begin(), result.end(), 0.0,
			[](double acc, std::pair<std::shared_ptr<Learn::EvaluationResult>, const TPG::TPGVertex*> pair)->double {return acc + pair.first->getResult(); });
		avg /= result.size();
		printf("%3d\t%4" PRIu64 "\t%1.2lf\t%1.2lf\t%1.2lf", i, la.getTPGGraph().getNbVertices(), min, avg, max);
		std::cout << "\t" << std::chrono::duration_cast<std::chrono::milliseconds>(stopEval - startEval).count();

#ifndef NO_CONSOLE_CONTROL
		generation = i;
		if (toggleDisplay) {
			bestRoot = iter->second;
			doDisplay = true;
			while (doDisplay);
		}
#endif
		std::cout.flush();

		if (!exitProgram) {
			auto startTrain = std::chrono::high_resolution_clock::now();
			la.trainOneGeneration(i);
			auto stopTrain = std::chrono::high_resolution_clock::now();

			std::cout << "\t" << std::chrono::duration_cast<std::chrono::milliseconds>(stopTrain - startTrain).count() << std::endl;
		}
	}

	// Keep best policy
	la.keepBestPolicy();
	dotExporter.setNewFilePath("out_best.dot");
	dotExporter.print();

	// cleanup
	for (unsigned int i = 0; i < set.getNbInstructions(); i++) {
		delete (&set.getInstruction(i));
	}

#ifndef NO_CONSOLE_CONTROL
	// Exit the thread
	std::cout << "Exiting program, press a key then [enter] to exit if nothing happens.";
	threadDisplay.join();
#endif

	return 0;
}