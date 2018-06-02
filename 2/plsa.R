##########################################################################################################################
#########Applying PLSA
##########################################################################################################################
require(dplyr)

#sampling
corp = tdm_f %>% as.matrix()
colSums(corp)
idx.word = rownames(corp) %>% unique #unique words

k = 5 #number of expectedd topics
n = corp%>%ncol #number of column
g = length(idx.word) #length of unique words


dt = foreach(i = 1:n,.combine = append) %do%{ #indicator for doc+word
  paste(i,idx.word)
}

posterior = matrix(runif(g*n*k,min = 0,max = 1),ncol = k)
colnames(posterior) = paste0("topic",1:k)
rownames(posterior) = dt

pz = matrix(rbeta(k,1,k), nc=k) %>% sweep(1, rowSums(.), FUN="/") #P(z) -> sum = 1
colnames(pz) = paste0("topic",1:k)

pi =  matrix(rbeta(n*k,k,n), nc=n) %>% sweep(1, rowSums(.), FUN="/") %>% t   #dimension = n * k(document number * topic number) 
rownames(pi) = paste0("docs",1:n)
colnames(pi) = paste0("topic",1:k)

pwz = matrix(rbeta(g*k,k,g), nc=g) %>% sweep(1, rowSums(.), FUN="/") %>% t #P(w|z)
rownames(pwz) = idx.word
colnames(pwz) = paste0("topic",1:k)


parameter = list(pz,pi,pwz)


#E-STEP : estimate posterior based on given random parameter


estep = function(parameter,posterior){
  res = posterior
  pz = parameter[[1]]
  pi = parameter[[2]]
  pwz = parameter[[3]]
  for(i in 1:k){ #topic loop
    
    for(j in 1:length(dt)){ #word * doc loop
      doc = unlist(strsplit(dt[j]," "))[1]
      word = unlist(strsplit(dt[j]," "))[2]
      
      res[dt[j],i] = pi[paste0("docs",doc),i] * pwz[word,i] / sum(pi[paste0("docs",doc),] * pwz[word,])
      
    }
    
    
  }
  
  
  return(res)
}



#M-STEP : estimate parameter based on calculated posterior in estep


mstep = function(parameter,posterior){
  pz = parameter[[1]]
  pi = parameter[[2]]
  pwz = parameter[[3]]
  
  for(i in 1:k){ #topic loop
    pz[1,i] = sum(posterior[,i]*corp) / sum(corp)
    for(j in 1:length(dt)){ #word * doc loop
      doc = unlist(strsplit(dt[j]," "))[1]
      word = unlist(strsplit(dt[j]," "))[2]
      
      pi[paste0("docs",doc),i] = sum(corp[,doc] * posterior[grep(doc,dt),i]) / sum(corp[,doc] * posterior[grep(doc,dt),])
      
      pwz[word,i] = sum(posterior[grep(word,dt),i]*corp[word,]) / sum(posterior[grep(word,dt),]*corp[word,])
    }
    
  }
  
  return(list(pz,pi,pwz))  
  
}

#Argmax P(C|^)

logpc = function(parameter,posterior){
  pz = parameter[[1]]
  pi = parameter[[2]]
  pwz = parameter[[3]]
  
  pc = matrix(0,ncol = n,nrow = g)
  rownames(pc) = idx.word
  colnames(pc) = 1:n
  for(i in 1:length(dt)){ #document and word loop
    doc = unlist(strsplit(dt[i]," "))[1]
    word = unlist(strsplit(dt[i]," "))[2]
    
    
    pc[word,doc] = corp[word,doc] * log(sum(pi[paste0("docs",doc),]*pwz[word,]))
    
    
  }
  
  return(sum(pc))
}



#iteration


iter = 100
posterior.iter = posterior
parameter.iter = parameter
res.posterior = list()
res.parameter = list()
res = c()
p = progress_estimated(iter)
for(i in 1:iter){
  res.posterior[[i]]  = estep(posterior = posterior.iter,parameter = parameter.iter)
  posterior.iter = res.posterior[[i]] #update posterior
  res.parameter[[i]] = mstep(posterior = posterior.iter,parameter = parameter.iter)
  parameter.iter = res.parameter[[i]] #update parameter
  res[i]  = logpc(parameter=parameter.iter,posterior = posterior.iter)
  
  
  p$tick()$print()
}

plot(res,type= "l",main = "log likelihood p(C|^)")



idx = which(res==max(res))

res.posterior[[idx]]

res.parameter[[idx]][[1]]
res.parameter[[idx]][[2]] 
res.parameter[[idx]][[3]] 