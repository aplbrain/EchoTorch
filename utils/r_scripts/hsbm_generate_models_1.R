#!/usr/bin/env Rscript

# Generate HSBMs models basic 

# M. Villafañe-Delgado 
# 11/29/2020

# Modified by:
# Sean McDaniel 2/9/2021







construct_hsbm_graph <- function() 
{
    curr_dir <- getwd()
    setwd(getwd())  # Set current working directory to be the working directory

    # Sample HSBM from https://igraph.org/r/doc/sample_hierarchical_sbm.html 
    # This example generates 10 blocks with 3 clusters in each block 

    n = 500                # Total number of vertices 
    m = 5                 # Total number of blocks. n/m must be an integer value 
    # rho = c(3, 3, 4)/10    # Either a list or a vector. Must contain the fraction of vertices per block and sum to 1 
    # rho = c(6, 6, 8)/20    # Either a list or a vector. Must contain the fraction of vertices per block and sum to 1 
    rho = c(2, 2, 1)/5    # Either a list or a vector. Must contain the fraction of vertices per block and sum to 1 


    C <- matrix(c(1  , 3/4,   0,
                3/4,   0, 3/4,
                0  , 3/4, 3/4), nrow=3)   # Bernoulli rates for within block clusters 
    p = 1/20               # Between block probability 

    # Generate HSBM 
    g <- sample_hierarchical_sbm(n, m, rho=rho, C=C, p=p)

    v <- 1    # Unique index or identifier for each network 

    # Save graph 
    fname <- sprintf("%s/graphs/hsbm_n_%s_m_%s_p_%s_v%s.graphml", curr_dir, n, m, p, v) 

    write_graph(g, fname, "graphml") 

    if (require(Matrix)) { image(g[]) }

    #Check if saved 
    gg <- read_graph(fname, "graphml")
    print(paste0('Graph successfully saved: ', !is.null(gg)))

}

construct_multiple_hsbm_graph <- function() 
{
    # Generate multiple HSBMs

    curr_dir <- getwd()
    setwd(getwd())  

    c = 3   # Total number of clusters per block - all blocks will have the same number of clusters

    # TODO: Add various probabilities 
    C <- matrix(c(1  , 3/4,   0,
                3/4,   0, 3/4,
                0  , 3/4, 3/4), nrow=3)   # Bernoulli rates for within block clusters 

    n_vals <- c(100, 500, 1000)
    m_vals <- c(5, 10, 20) 

    for(n in n_vals)     
    {
        for(m in m_vals)
        {

            # For the current 3 clusters per block case 
            if (m == 5) 
            {
                rho = c(2, 2, 1)/m 
            } 
            else if (m == 10) 
            {
                rho = c(3, 3, 4)/m
            } 
            else if (m == 20) 
            {
                rho = c(6, 6, 8)/m 
            }
        
            for(i_p in 1:20)
            {
                p = runif(1)               # Between block probability 

                # Generate HSBM 
                g <- sample_hierarchical_sbm(n, m, rho=rho, C=C, p=p)

                # Save graph 
                fname <- sprintf("%s/graphs/hsbm_%s_clusters_n_%s_m_%s_p_%#.3f.graphml", curr_dir, c, n, m, p) 
            
                write_graph(g, fname, "graphml")

                #Check if saved 
                #gg <- read_graph(fname, "graphml")
                #print(paste0('Graph successfully saved: ', !is.null(gg)))
            }
        }
    }
}

main <- function()
{
    library(igraph)
    
    #construct_hsbm_graph()
    construct_multiple_hsbm_graph()
}

main()