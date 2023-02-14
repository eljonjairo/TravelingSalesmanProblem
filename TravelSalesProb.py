#!/usr/bin/env python3

# Genetic Algorithm to solve Traveling salesman problem
# 
# John Diaz january 2023
#TO DO List:
#  Cities Map
#

# import libraries
import numpy as np 
import random
from scipy.spatial import distance
import matplotlib.pyplot as plt

# Declaring global variables
global ncity
global npop
global nelit

# Input Parameters
Longx = 100                            # Horizontal length of map
Longy = 50                            # Vertical length of map

ncity = 6                              # Number of cities 

npop = 50                           # Size of population
ngen = 100                             # Generations number
mutProb = 0.1                        # Probablity of mutation
eliPort = 0.2                         # Portion of elitism

# Calculate number of elite models
nelit = int(npop*eliPort)

# Definition of functions

def initCityCoord(Longx,Longy):
    # Initialize x,y coordinates array for cities
    coord = np.zeros([ncity,2])
    
    # Generate random coordinates
    for icity in range (ncity):
        coord[icity,0] = random.uniform(-Longx/2,Longx/2)
        coord[icity,1] = random.uniform(-Longy/2,Longy/2)
    
    return coord    

def initPopulation():
   
    Population = np.zeros([npop,ncity],dtype=np.int8)
    scity = list(range(ncity))
    for ipop in range(npop):
        random.shuffle(scity)
        Population[ipop,] = scity
              
    return Population
 
def CrossOver(Population):
    # Calculate the middle of the individual
    nc = ncity//2
  
    for ipop in range(nelit,npop,2):
        PopAux1 = [] 
        # Save the whole indviudal starting in nc
        for ip in Population[ipop,0:nc]:
            PopAux1.append(int(ip))
        
        for ip in Population[ipop+1,nc:]:
            PopAux1.append(int(ip))
        
        for ip in Population[ipop+1,0:nc]:
            PopAux1.append(int(ip))
        
        PopAux1 = sorted(set(PopAux1), key=PopAux1.index)
        PopAux2 = [] 
        # Save the whole indviudal starting in nc
        for ip in Population[ipop+1,0:nc]:
            PopAux2.append(int(ip))
        
        for ip in Population[ipop,nc:]:
            PopAux2.append(int(ip))
        
        for ip in Population[ipop,0:nc]:
            PopAux2.append(int(ip))
        
        PopAux2 = sorted(set(PopAux2), key=PopAux2.index)
        
        Population[ipop] = np.array(PopAux1)
        Population[ipop+1] = np.array(PopAux2)
    
    return Population
    
def Mutation(Population,mutProb):
    for ipop in range(npop):  
        mutation = random.random()
        if mutation < mutProb:
           
            imut  = np.random.randint(low=0, high=ncity, size=2)
            city1 = int(Population[ipop,imut[0]])
            Population[ipop,imut[0]] = Population[ipop,imut[1]]
            Population[ipop,imut[1]] = city1
            
        return Population    
 
def evalMsfit(Population,Coord):
    Msfit = np.zeros((npop,))
       
    for ipop in range (npop):

        dist = 0
        for icity in range (ncity-1):
            
            iniCity = [Coord[Population[ipop,icity],0],Coord[Population[ipop,icity],1]]
            finCity = [Coord[Population[ipop,icity+1],0],Coord[Population[ipop,icity+1],1]]
            dist +=  distance.euclidean(iniCity,finCity) 
                    
        iniCity = [Coord[Population[ipop,-1],0],Coord[Population[ipop,-1],1]]
        finCity = [Coord[Population[ipop,0],0],Coord[Population[ipop,0],1]]
        dist +=  distance.euclidean(iniCity,finCity) 
   
        Msfit[ipop] = 1/dist
    
    dist = 1/Msfit    
    # Sort the Population and Misfit from best to worst model
    Population = Population[np.argsort(dist)]
    Msfit = Msfit[np.argsort(dist)]
    
    return Msfit    

def Selection(Population,Msfit):
   
    PopAux = Population
    # Reinitialize Population including nelit best individuals from previous generation
    Population = np.zeros([npop,ncity],dtype=np.int8)
    Population[0:nelit] = PopAux[0:nelit]

    # Build the roulette with slide proportional to the individual msfit
    TotalMsfit = Msfit/(sum(Msfit))
  
    Roulette =  np.zeros((npop,))
    Roulette[0] = TotalMsfit[0]
    for ipop in range(1,npop):
         Roulette[ipop] = Roulette[ipop-1]+TotalMsfit[ipop]
    
    # run the roulette npop-nelit times to fill the new generation population
    for ipop in range(nelit,npop):
    # roll the dices
        value = random.random()
        Raux  = Roulette[np.where(Roulette<=value)]
        naux = int(Raux.size)
       
        if naux == 0:
             index = 0
        else:
             idex = np.where(Roulette == Raux[-1])
             index = idex[0][0]
   
        Population[ipop] = PopAux[index]

    return Population

    
# Main part of the code

print("  ")
print(" START PROGRAM ")
print("  ")

print(" Generate Cities x,y Coordinates...")
print("  ")
# Generate x,y coordinates for cities
Coord = initCityCoord(Longx,Longy)

fig = plt.figure()
plt.plot(Coord[:,0],Coord[:,1],'o',color='k')
for icity in range(ncity):
    plt.text(Coord[icity,0],Coord[icity,1]+2,str(icity))

plt.xlim([-Longx/2-10, Longx/2+10])
plt.ylim([-Longy/2-10, Longy/2+10])

print(" Initialize Population...")
print("  ")
# Generate initial population
Population = initPopulation()   
    
print(" Starting Evolution Process...")
print("  ")
for igen in range (ngen):
    print(f" Generation {igen+1}  of  {ngen}")
    
    print(f"    Crossfit process...")
    Population = CrossOver(Population)
   
    print(f"    Mutation process...")
    Population = Mutation(Population,mutProb)
    
    print(f"    Selection process...")
    Msfit = evalMsfit(Population,Coord)
 
    Population = Selection(Population,Msfit)
    print("  ")
    
    print(f"    Best travel: {Population[0]} ")