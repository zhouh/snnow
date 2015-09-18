/*************************************************************************
	> File Name: src/depparser/BeamDecodor.h
	> Author: Hao Zhou
	> Mail: haozhou0806@gmail.com 
	> Created Time: 18/09/15 17:03:00
 ************************************************************************/

#include "Beam.h"
#include "State.h"

class BeamDecodor{


    
public:
    Beam beam;
    State * lattice;
    State * lattice_index[maxRound];
    State * correctState;
    int nMaxLatticeSize;
    int nRound
   
    BeamDecodor(int maxRound, int beamSize) : beam(beamSize) {

        nMaxLatticeSize =  (beamSize + 1) * maxRound;
        nRound = 0;

        // construct the lattice
        lattice = new State[mnMaxLatticeSize];
        lattice_index[maxRound];
        correctState = lattice;
    }

    ~BeamDecodor(){
        delete[] lattice;

    }

};
