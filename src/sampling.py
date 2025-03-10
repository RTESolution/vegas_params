from .base import Parameter
import numpy as np
#adds  sampling method to the parameter
def _sample_with_normalized_factor(self:Parameter, size=1, iter_max=10):
        """Generate sample of the given size, but applying the factor as survival probability, 
        i.e. randomly dropping the values with low factor.

        Note: this process is iterative (it might need to generate several samples), and several times slower than regular :meth:`sample` method
        """
        #start with the sample of size N
        selected = []
        N = size
        N_generate = int(N*2) #the size of next sample to generate
        for it in range(iter_max):
            sample, factor = self.sample(N_generate), self.factor
            #expand factor
            factor = factor*np.ones(shape=(N_generate))
            if it==0:
                if np.min(self.factor)==np.max(self.factor):
                    #the factor is equal for all values, so just return this sample
                    return sample[:N]
                    
            random = np.random.uniform(size=len(sample))
            if it==0:
                the_sample = sample
                the_factor = factor
                the_random = random
            else:
                the_sample = np.append(the_sample, sample, axis=0)
                the_factor = np.append(the_factor, factor, axis=0)
                the_random = np.append(the_random, random, axis=0)
            #apply the selection
            selected = (the_random*the_factor.max()) < the_factor
            N_selected = selected.sum()
            print(f"Iteration #{it}: generated {N_generate} -> selected={N_selected}/{len(the_factor)}")
            if N_selected>=N:
                self.factor = np.ones(N)*N/len(the_factor)
                return the_sample[selected][:N]
            else:
                prob = N_selected/len(the_sample) #estimated selection probability
                #now decide how many items to generate
                # N = prob*(N_total) = N_selected + prob*N_generate =>
                N_generate = (N-N_selected)/prob
                #additional 20% to avoid making small iterations
                N_generate *= 1.2
                #make sure we don't make a sample too small or too large
                N_generate = np.clip(N_generate, N/100, N*1000)
                N_generate = np.asarray(np.ceil(N_generate), dtype=int)
        raise RuntimeError(f"Maximum iterations ({iter_max}) reached. Generated only {sum(selected)} points of {N} requested")

Parameter.sample_with_factor = _sample_with_normalized_factor