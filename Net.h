#ifndef NET_H
#define NET_H

#include <iostream>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include "Neuron.h"

class Net
{
public:

	/*
	    You should *not* change this part
	*/

	// constructor. 
	// topology is a container representing net structure. 
	//   e.g. {2, 4, 1} represents 2 neurons for the first layer, 4 for the second layer, 1 for last layer
	// if you want to hard-code the structure, just ignore the variable topology 
	// eta: learning rate 
	Net(const std::vector<unsigned> &topology, const double eta);

	// given an input sample inputVals, propagate input forward, compute the output of each neuron 
	void feedForward(const std::vector<double> &inputVals);

	// given the vector targetVals (ground truth of output), propagate errors backward, and update each weights
	void backProp(const std::vector<double> &targetVals);

	// output the prediction for the current sample to the vector resultVals
	void getResults(std::vector<double> &resultVals) const;

	// return the error of the current sample
	double getError(void) const;

	
	/*
	    Add what you need in the below
	*/


	// ...

private:
	// ...
	// const double eta;
	// const int numInputUnit;
	// const int numHiddenUnit;
	// const int numOutputUnit;
	// sigmoid function
	// double sigmoid(double x) const;

	// // weight between layer i, jth neuron and layer i+1, kth neuron; weight[i][j][k] 
	// double ***weight;

	// // output of neuron 
	// double ** output;

	// // the number of neurons in each layer
	// int* numNeurons;

	std::vector<Layer> m_layers; //m_layers[layerNum][neuronNum]
	double m_error;
	// double m_recentAverageError;
	// double m_recentAverageSmoothingFactor;

};




#endif//NET_H
