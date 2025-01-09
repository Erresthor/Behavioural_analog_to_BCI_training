


import jax.numpy as jnp

if __name__ == "__main__":
    
    Nu = 3
    Ns = 5
    
    vect = jnp.array([0.5,0.2,0.0])
    
    
    new_vect=  jnp.repeat(jnp.expand_dims(vect,-1),Ns,-1)
    print(new_vect)
    print(new_vect.shape)