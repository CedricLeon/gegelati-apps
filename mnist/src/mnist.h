#ifndef MNIST_H
#define MNIST_H

#include <random>

#include <gegelati.h>

#include "mnist_reader/mnist_reader.hpp"

/**
* LearningEnvironment to train an agent to classify the MNIST database.
*/
class MNIST : public Learn::ClassificationLearningEnvironment {
protected:
	/// MNIST dataset for the training.
	static mnist::MNIST_dataset<std::vector, std::vector<double>, uint8_t> dataset;

	/// Current LearningMode of the LearningEnvironment.
	Learn::LearningMode currentMode;

	/// Randomness control
	Mutator::RNG rng;

	/// Current image provided to the LearningAgent
	Data::PrimitiveTypeArray2D<double> currentImage;

	/// Current index of the image in the dataset.
	uint64_t currentIndex;

    /// Seed of the training
    size_t seed;

	/**
	* \brief Change the image currently available in the dataSources of the LearningEnvironment.
	*
	* A random image from the dataset for the current mode is selected.
	*
	*/
	void changeCurrentImage();

public:

	/**
	* \brief Constructor.
	*
	* Loads the dataset on construction. Sets the LearningMode to TRAINING.
	* Sets the score to 0.
	*/
	MNIST(size_t seed);

	/// Inherited via LearningEnvironment
	virtual void doAction(uint64_t actionID) override;

	/// Inherited via LearningEnvironment
	virtual void reset(size_t seed = 0, Learn::LearningMode mode = Learn::LearningMode::TRAINING) override;

	/// Inherited via LearningEnvironment
	virtual std::vector<std::reference_wrapper<const Data::DataHandler>> getDataSources() override;

	/// Inherited via LearningEnvironment
	virtual bool isCopyable() const override;

	/// Inherited via LearningEnvironment
	virtual LearningEnvironment* clone() const;

	/**
	* Get the score of the current evaluation session (i.e. since the last reset).
	*
	* For the MNIST LearningEnvironment, the score is computed as follows:
	* - Score is incremented by 1.0 for a correct classification.
	* - Score is decremented by 0.25 for an incorrect classification.
	* - Score is left unchanged for action ID 10 which corresponds to an
	*   absence of guess from the agent.
	*
	* To be used in a viable learning process, score can be compared only on
	* agent evaluated on the same number of samples.
	*/
	virtual double getScore() const override;

	/**
	* \brief This classification LearningEnvironment will never reach a
	* terminal state.
	*
	* \return false.
	*/
	virtual bool isTerminal() const override;

	/**
	* \brief Retrieve the label of the current image.
	*
	* Should not be used to cheat.
	*/
	uint8_t getCurrentImageLabel();

	/**
	* \brief Function printing the statistics of the classification.
	*
	* \param[in] result the Map containing the list of roots within a TPGGraph,
	* with their score in ascending order.
	*/
	void printClassifStatsTable(const Environment& env, const TPG::TPGVertex* bestRoot, const int numGen, std::string const& outputFile, bool readable);
};

#endif
