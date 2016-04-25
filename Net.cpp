#include "Net.h"

Net::Net(const std::vector<unsigned> &topology, const double eta){
   std::cout << "net constructor" << std::endl; 
}

void Net::feedForward(const std::vector<double> &inputVals) {
   std::cout << "feed forward" << std::endl;
}

void Net::backProp(const std::vector<double> &targetVals) {
   std::cout << "back prop" << std::endl;
}

void Net::getResults(std::vector<double> &resultVals) const {
    std::cout << "result" << std::endl;
}

double Net::getError(void) const {
    std::cout << "error" << std::endl;
}
