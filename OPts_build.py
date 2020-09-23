# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 20:16:01 2020

@author: Naveen Raj
"""
import numpy as np

class gen_ops():
    def crossover(x,y):
        r=np.random.random(1)
        a1=r*x+(1-r)*y
        a2=r*y+(1-r)*x
        if np.random.random(1) > 0.5:
            return a1
        else:
            return a2
            
    def mutation(x,mut):
        if np.random.random(1) <= mut:
            return x-np.random.uniform(-1,1)*x
        else:
            return x
    
    def roulette_wheel(P):
        y=np.cumsum(P)
        return np.where(y > np.random.random(1))[0][0]

###====================Genetic Algorithm--- STARTS------=======================================
def GA(fun,nvar,lb,ub,A,B,Aeq,Beq,nolncon,maxiter,numpart):
    
    class population():
        def __init__(self,x):
            self.position=x
            #self.cost=fun(x)
        def cost_eval(self,fun):
            self.cost=fun(self.position)
        
        def apply_bound(self,lb,ub):#,vmax,vmin):
            self.postion=np.minimum(self.position,ub)
            self.position=np.maximum(lb,self.position)

    pop_size=numpart
    pop=[]
    elit=0.25
    #cross=1-elit
    mut=0.10
    no_elit=pop_size*elit
    #no_cross=pop_size-no_elit

    for i in range(pop_size):
        pop.append(population(np.random.uniform(lb,ub)))#,function)
        best=[]
    for i in range(maxiter):
        cost_ranked=[]
        pops_corssover=[]
        pop_nxt=[]
        cost=[]
    
        for pops in pop:
            pops.apply_bound(lb,ub)
            pops.cost_eval(fun)
            cost.append(pops.cost)
        
        Rank=np.argsort(cost)
        [cost_ranked.append(pop[k].cost) for k in Rank[1:60]]
        
        [pops_corssover.append(pop[k]) for k in Rank[1:60]]
            
        cost_ranked_p=cost_ranked/np.sum(cost_ranked)
    
        [pop_nxt.append(pop[k]) for k in Rank[1:int(no_elit)]]
    
        for j in range(int(no_elit),pop_size+1):
    
            y1=gen_ops.roulette_wheel(cost_ranked_p)
            y2=gen_ops.roulette_wheel(cost_ranked_p)
            child=gen_ops.crossover(pops_corssover[y1].position,pops_corssover[y2].position)
            child=gen_ops.mutation(child,mut)
            pop_nxt.append(population(child))

        pop=pop_nxt
        best.append(np.min(cost))
    return best[maxiter-1]
###====================Genetic Algorithm--- Ends------=======================================
        
###====================PSO--- STARTS------=======================================
   
def PSO(fun,nvar,lb,ub,A,B,Aeq,Beq,noln,maxiter,numpart):  
    class Particle():
        def __init__(self,x,v):
            self.position = x
            self.velocity = v
            self.best_value = 1e6
            self.best_position = self.position
            self.value = 0
        
        def get_value(self, fun):

            value = fun(self.position)
            self.value = value
            if value < self.best_value:   # minimisation option
                self.best_value = value
                self.best_position = self.position
        

        def update_position(self, dt=1):

            self.position += np.array(self.velocity) * dt
    
        def update_velocity(self, global_best, coefficients):

            self.velocity = self.velocity * coefficients[0] + \
            coefficients[1] * np.multiply(np.random.random(len(self.position)),(self.best_position - self.position)) + \
            coefficients[2] * np.multiply(np.random.random(len(self.position)),(global_best - self.position))
                
        def apply_bound(self,lb,ub,vmax,vmin):
            self.postion=np.minimum(self.position,ub)
            self.position=np.maximum(lb,self.position)
            self.velocity=np.minimum(self.velocity,vmax)
            self.velocity=np.maximum(vmin,self.velocity)
        
    pop_size=numpart


    particles = []

    vmax=(ub-lb)
    vmin=-vmax
# =============================================================================
#    coefficients=np.zeros(3)
#    coefficients[0]=1.5            # Inertia Weight
#    wdamp=0.99     # Inertia Weight Damping Ratio
#    coefficients[1]=1.5         # Personal Leopearning Coefficient
#    coefficients[2]=2.0         # Global Learning Coefficient
# 
# =============================================================================

# Constriction Coefficients
    coefficients=np.zeros(3)
    phi1=2.1
    phi2=2.1
    phi=phi1+phi2
    chi=2/(phi-2+np.sqrt(phi**2-4*phi))
    coefficients[0]=chi          # Inertia Weight
    wdamp=1        # Inertia Weight Damping Ratio
    coefficients[1]=chi*phi1    # Personal Learning Coefficient
    coefficients[2]=chi*phi2    # Global Learning Coefficient


    for i in range(pop_size):
        particles.append(Particle(np.random.uniform(lb,ub),np.random.uniform(vmax,vmin)))

    
    best_value=np.zeros(maxiter)
    for i in range(maxiter):
        values = []

        for particle in particles:  # calculation loop
            particle.apply_bound(lb,ub,vmax,vmin)
            particle.get_value(fun)
            values.append(particle.value)
        
        if i == 0:
            best_value[i]=min(values)
            gbest_pop=particles[values.index(min(values))].position
        else:
            best_value[i]=fun(gbest_pop)
            if min(values) < best_value[i]:
                best_value[i] = min(values)
                gbest_pop=particles[values.index(min(values))].position


        for particle in particles:
            particle.update_velocity(gbest_pop, coefficients)
            particle.apply_bound(lb,ub,vmax,vmin)
            particle.update_position()
            
        coefficients[0]=wdamp*coefficients[0];    
    
    return best_value[maxiter-1]
###====================PSO--- ENDS------=======================================

###====================QPSO--- STARTS----=======================================
    
def QPSO(fun,nvar,lb,ub,A,B,Aeq,Beq,noln,maxiter,numpart):
    class Particle():
        def __init__(self,x,v):
            self.position = x
            self.velocity = v
            self.best_value = 1e6
            self.best_position = self.position
            self.value = 0
    
        def get_value(self, fun):
            """Method getting value of function fun in current position
                Keyword arguments:
                fun -- function defined as other method
                """
                value = fun(self.position)
                self.value = value
                if value < self.best_value:   # minimisation option
                    self.best_value = value
                    self.best_position = self.position
        
        def update_position(self, alpha=0.75):
            """Method updating position of particle in current moment
            Keyword arguments:
                dt -- time step, default=1
                """
        
            u=np.random.random(self.position.shape)
            if np.random.random(1) <= 0.5:
                self.position =self.best_position+alpha*np.abs(self.position-self.best_position)*np.log(1/u)
            else:
                self.position =self.best_position-alpha*np.abs(self.position-self.best_position)*np.log(1/u)
            
        def update_velocity(self, global_best, coefficients):
            """Method updating velocity of particle in current moment
            Keyword arguments:
                global_best -- double[d] current best position of population, d-dimensions
                coefficients -- double[3] vector of three coefficients of simulation: 0-inertial 1-egoism 2-group terms
                """
        #for i in range(len(global_best)):
            self.velocity = self.velocity * coefficients[0] + \
            coefficients[1] * np.multiply(np.random.random(len(self.position)),(self.best_position - self.position)) + \
            coefficients[2] * np.multiply(np.random.random(len(self.position)),(global_best - self.position))
            phi=np.random.random(self.position.shape)
            self.best_position=phi*self.best_position+(1-phi)*global_best
                
        def apply_bound_V(self,lb,ub,vmax,vmin):
        #self.postion=np.minimum(self.position,ub)
        #self.position=np.maximum(lb,self.position)
            self.velocity=np.minimum(self.velocity,vmax)
            self.velocity=np.maximum(vmin,self.velocity)
        
        def apply_bound_P(self,lb,ub,vmax,vmin):
            self.postion=np.minimum(self.position,ub)
            self.position=np.maximum(lb,self.position)
            #self.velocity=np.minimum(self.velocity,vmax)
            #self.velocity=np.maximum(vmin,self.velocity)
        

# =============================================================================
# def function(x):
#     return np.sum(np.square(x))
# nvar=10;
# lb=-100*np.ones(nvar)
# ub=100*np.ones(nvar)
# #x_end = y_end = 8
# =============================================================================

    #maxiter=300
    pop_size=numpart


    particles = []
    
    vmax=(ub-lb)
    vmin=-vmax

    # Constriction Coefficients
    coefficients=np.zeros(3)
    phi1=2.1
    phi2=2.1
    phi=phi1+phi2
    chi=2/(phi-2+np.sqrt(phi**2-4*phi))
    coefficients[0]=chi          # Inertia Weight
    wdamp=1        # Inertia Weight Damping Ratio
    coefficients[1]=chi*phi1    # Personal Learning Coefficient
    coefficients[2]=chi*phi2    # Global Learning Coefficient
    
    
    for i in range(pop_size):
        particles.append(Particle(np.random.uniform(lb,ub),np.random.uniform(vmax,vmin)))
        
    
    best_value=np.zeros(maxiter)
    for i in range(maxiter):
        values = []
        #    ax = fig.add_subplot(111, projection='3d')
        
        for particle in particles:  # calculation loop
            particle.apply_bound_P(lb,ub,vmax,vmin)
            particle.get_value(function)
            values.append(particle.value)
            
            if i == 0:
                best_value[i]=min(values)
                gbest_pop=particles[values.index(min(values))].position
            else:
                best_value[i]=function(gbest_pop)
                if min(values) < best_value[i]:
                    best_value[i] = min(values)
                    gbest_pop=particles[values.index(min(values))].position
                    #for particle in particles:  # update loop
        for particle in particles:
            particle.update_velocity(gbest_pop, coefficients)
            particle.apply_bound_V(lb,ub,vmax,vmin)
            particle.update_position()
        #particle.apply_bound(lb,ub,vmax,vmin)
            
        coefficients[0]=wdamp*coefficients[0];
        
    return best_value[maxiter-1]   