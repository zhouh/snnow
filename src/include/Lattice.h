/*************************************************************************
	> File Name: src/include/Lattice.h
	> Author: Hao Zhou
	> Mail: haozhou0806@gmail.com 
	> Created Time: 08/09/15 13:39:13
 ************************************************************************/

#ifndef LATTICE_H_
#define LATTICE_H_

#include "State.h"

class Lattice{

    int m_nMaxLatticeSize;
    int m_nMaxRound;
    State * m_lattice = new State[max_lattice_size];
    State * m_lattice_index[maxRound];

    Lattice(int m_nMaxRound, int m_nMaxLatticeSize){
        
        this->m_nMaxLatticeSize = m_nMaxLatticeSize;
        this->m_nMaxRound = m_nMaxRound;
        m_lattice = new State[m_nMaxLatticeSize];

    }

    ~Lattice(){
        delete m_lattice;
    }
    
};

#endif


