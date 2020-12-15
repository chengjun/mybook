library(RSiena)

?RSiena

# Now we use the internally available s50 data set.
# Look at its description:
# 3 waves, 50 actors
?s50

head(s501)

tail(s501)

# and at the alcohol variable
head(s50a)

# Now define the objects with the same names as above
# (this step is superfluous if you read the data already).
friend.data.w1 <- s501
friend.data.w2 <- s502
friend.data.w3 <- s503
drink <- s50a
smoke <- s50s

# Now the data must be given the specific roles of variables
# in an RSiena analysis.
#
# Dependent variable

?sienaDependent

# First create a 50 * 50 * 3 array composed of the 3 adjacency matrices
friendshipData <- array( c( friend.data.w1, friend.data.w2, friend.data.w3 ),
           dim = c( 50, 50, 3 ) )

# and next give this the role of the dependent variable:

friendship <- sienaDependent(friendshipData)
# What did we construct?
friendship

# We also must prepare the objects that will be the explanatory variables.
#
# Actor covariates
# We use smoking for wave 1 as a constant actor covariate:
smoke1 <- coCovar( smoke[ , 1 ] )
# A variable actor covariate is defined for drinking:
alcohol <- varCovar( drink )
# (This choice is purely for the purpose of illustration here.)

# Put the variables together in the data set for analysis
?sienaDataCreate

mydata <- sienaDataCreate( friendship, smoke1, alcohol )
# Check what we have
mydata

# You can get an outline of the data set with some basic descriptives from
print01Report( mydata, modelname="./data/s50_RSiena_output")

# For the model specification we need to create the effects object
myeff <- getEffects( mydata )
# All the effects that are available given the structure
# of this data set can be seen from
effectsDocumentation(myeff)

?effectsDocumentation

# For a precise description of all effects, see Chapter 12 in the RSiena manual.
# A basic specification of the structural effects:
?includeEffects

myeff <- includeEffects( myeff, transTrip, cycle3)
# and some covariate effects:
myeff <- includeEffects( myeff, egoX, altX, simX, interaction1 = "alcohol" )
myeff <- includeEffects( myeff, simX, interaction1 = "smoke1" )
myeff

# Create object with algorithm settings
# Accept defaults but specify name for output file
# (which you may replace by any name you prefer)
#?sienaAlgorithmCreate
myalgorithm <- sienaAlgorithmCreate( projname = './data/s50_RSiena_output' )

# Estimate parameters
#?siena07
ans <- siena07( myalgorithm, data = mydata, effects = myeff)
ans

# This gives results from a random starting point.
# To use a fixed starting point, use the "seed" parameter:
# myalgorithm <- sienaAlgorithmCreate( projname = 's50', seed=435123 )

# For checking convergence, look at the
# 'Overall maximum convergence ratio' mentioned under the parameter estimates.
# It can also be shown by requesting
ans$tconv.max
# If this is less than 0.25, convergence is good.
# If convergence is inadequate, estimate once more,

# using the result obtained as the "previous answer"
# from which estimation continues:

ans <- siena07( myalgorithm, data = mydata, effects = myeff, prevAns=ans)
ans

# If convergence is good, you can look at the estimates.
# More extensive results
summary(ans)

# Still more extensive results are given in the output file
# s50.txt in the current directory.

# Note that by putting an R command between parentheses (....),
# the result will also be printed to the screen.

# Next add the transitive reciprocated triplets effect,
# an interaction between transitive triplets and reciprocity,

(myeff <- includeEffects( myeff, transRecTrip))
(ans1 <- siena07( myalgorithm, data = mydata, effects = myeff, prevAns=ans))
# If necessary, repeat the estimation with the new result:
(ans1 <- siena07( myalgorithm, data = mydata, effects = myeff, prevAns=ans1))

# This might still not have an overall maximum convergence ratio
# less than 0.25. If not, you could go on once more.
#
# Inspect the file s50.txt in your working directory
# and understand the meaning of its contents.


# To have a joint test of the three effects of alcohol:
# ?Multipar.RSiena
Multipar.RSiena(ans1, 7:9)
# Focusing on alcohol similarity, the effect is significant;
# diluting the effects of alcohol by also considering ego and alter,
# the three effects simultaneously are not significant.


###                Assignment 1                                              
# 1a.
# Drop the effect of smoke1 similarity and estimate the model again.
# Do this by the function setEffects() using the <<include>> parameter.
# Give the changed effects object and the new answer object new names,
# such as effects1 and ans1, to distinguish them.
# 1b.
# Change the three effects of alcohol to the single effect
# of alcohol similarity, and estimate again.


###                Networks and behavior study                               
# Now we redefine the role of alcohol drinking
# as a dependent behaviour variable.
# Once again, look at the help file
# ?sienaDependent
# now paying special attention to the <<type>> parameter.
drinking <- sienaDependent( drink, type = "behavior" )

# Put the variables together in the data set for analysis
NBdata <- sienaDataCreate( friendship, smoke1, drinking )
NBdata

NBeff <- getEffects( NBdata )
effectsDocumentation(NBeff)
NBeff <- includeEffects( NBeff, transTrip, transRecTrip )
NBeff <- includeEffects( NBeff, egoX, egoSqX, altX, altSqX, diffSqX,
                         interaction1 = "drinking" )
NBeff <- includeEffects( NBeff, egoX, altX, simX, interaction1 = "smoke1" )
NBeff

# For including effects also for the dependent behaviour variable, see
# ?includeEffects
NBeff <- includeEffects( NBeff, avAlt, name="drinking",
                         interaction1 = "friendship" )
NBeff

# Define an algorithm with a new project name
myalgorithm1 <- sienaAlgorithmCreate( projname = './data/s50NB_RSiena_output' )

# Estimate again, using the second algorithm right from the start.
NBans <- siena07( myalgorithm1, data = NBdata, effects = NBeff)
# You may improve convergence (considering the overall maximum
# convergence ratio) by repeated estimation in the same way as above.

# Look at results
NBans

# Make a nicer listing of the results
siena.table(NBans, type="html", sig=TRUE)
# This produces an html file; siena.table can also produce a LaTeX file.
# But the current version (2.12) does not do this well for html.

###                Assignment 2                                              ###
################################################################################

# 2a.
# Replace the average alter effect by average similarity (avSim)
# or total similarity (totSim) and estimate the model again.
# 2b.
# Add the effect of smoking on drinking and estimate again.

################################################################################

################################################################################
###                Assignment 3                                              ###
################################################################################

# Read Sections 13.3 and 13.4 of the Siena Manual, download scripts
# SelectionTables.r and InfluenceTables.r from the Siena website,
# and make plots of the selection table and influence table for drinking.



