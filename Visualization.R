setwd("~/Documents/GitHub/Personal Projects/Research/Parsons Puzzles - Amruth/Complete")

library("TraMineR")
library("stringr")

require("pscl") # alternatively can use package ZIM for zero-inflated models
library("lmtest")
library("MASS")
library("rsq")

virtualization_seq_freq <- function(df){

cl1.3 <- df$Clusters
max.cluster <- max(cl1.3)
temp_2 <- strsplit(df$new_sequence, "")
type_label <- sprintf("Type %d",1:max.cluster)
max.length <- max(sapply(temp_2, length))
temp_2 <- lapply(temp_2, function(v) { c(v, rep(NA, max.length-length(v)))})
temp <- data.frame(do.call(rbind, temp_2))

temp.alphabet <- c("T","P","O","I","V","F","E", "R");
temp.labels <- c("Trash","Braces","Output","Input","Variable","If","Else", "Reorder")
temp.scodes <- c("T","P","O","I","V","F","E", "R")
data_1_E.seq <- seqdef(temp, 0:max.length, alphabet = temp.alphabet, states=temp.scodes, labels=temp.labels)

cpal(data_1_E.seq) <- c("#7FC97F", "#BEAED4", "#FDC086", "#FFFF99", "#386CB0", "#F0027F", "#BF5B17", "#666666")

#idxs used to find interval representing the range to be plotted.
seqlegend(data_1_E.seq) 
seqiplot(data_1_E.seq, with.legend=F, main = "First 10 Sequence" ) 
seqfplot(data_1_E.seq, pbarw=T, with.legend=F, main = "Most Frequent Sequences") 
seqdplot(data_1_E.seq, with.legend=F,  main = "State Distribution") 

cl1.3fac <- factor(cl1.3, labels = type_label)
# 
# seqdplot(data_1_E.seq, group=cl1.3fac,main = "Type State Distribution", border=NA)
seqfplot(data_1_E.seq, group=cl1.3fac, pbarw=T, main = "Most Frequent Sequences")

}

virtualization_seq_dist <- function(df){
  
  cl1.3 <- df$Clusters
  max.cluster <- max(cl1.3)
  temp_2 <- strsplit(df$final_sequence, "")
  type_label <- sprintf("Type %d",1:max.cluster)
  max.length <- max(sapply(temp_2, length))
  temp_2 <- lapply(temp_2, function(v) { c(v, rep(NA, max.length-length(v)))})
  temp <- data.frame(do.call(rbind, temp_2))
  
  temp.alphabet <- c("T","P","O","I","V","F","E", "R", "X");
  temp.labels <- c("Trash","Braces","Output","Input","Variable","If","Else", "Reorder", "Padding")
  temp.scodes <- c("T","P","O","I","V","F","E", "R", "X")
  data_1_E.seq <- seqdef(temp, 0:max.length, alphabet = temp.alphabet, states=temp.scodes, labels=temp.labels)
  
  cpal(data_1_E.seq) <- c("#7FC97F", "#BEAED4", "#FDC086", "#FFFF99", "#386CB0", "#F0027F", "#BF5B17", "#666666", "white")
  
  #idxs used to find interval representing the range to be plotted.
  # seqlegend(data_1_E.seq) 
  # seqiplot(data_1_E.seq, with.legend=F, main = "First 10 Sequence" ) 
  # seqfplot(data_1_E.seq, pbarw=T, with.legend=F, main = "Most Frequent Sequences") 
  # seqdplot(data_1_E.seq, with.legend=F,  main = "State Distribution") 
  
  cl1.3fac <- factor(cl1.3, labels = type_label)
  
  seqdplot(data_1_E.seq, group=cl1.3fac,main = "Type State Distribution", border=NA)
  # seqfplot(data_1_E.seq, group=cl1.3fac, pbarw=T, main = "Most Frequent Sequences")
  
}

par(mfrow=c(2,2))
df_new <- read.csv("Puzzles_Output/standard_best_final_output.csv")
df_new$new_sequence <- as.character(df_new$new_sequence)
df_new$final_sequence <- as.character(df_new$new_sequence)

max_char <- max(nchar(df_new$final_sequence))

for(x in 1:nrow(df_new)){
  padding <- 0
  seq_len <- nchar(df_new$final_sequence[x])
  if(seq_len != max_char){
    padding = max_char - seq_len
    while(padding != 0){
      df_new$final_sequence[x] <- paste(df_new$final_sequence[x], "X", sep="")
      padding = padding-1
    }
  }
}

virtualization_seq_freq(df_new)
virtualization_seq_dist(df_new)




