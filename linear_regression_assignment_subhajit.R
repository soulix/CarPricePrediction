rm(list=ls())
install.packages("MASS",repos="http://lib.stat.cmu.edu/R/CRAN")
install.packages("car",repos="http://lib.stat.cmu.edu/R/CRAN")
install.packages("caret",repos="http://lib.stat.cmu.edu/R/CRAN")
install.packages("PerformanceAnalytics")
install.packages("corrplot")
install.packages("lazyeval")
library(MASS)
library(car)
library(ggplot2)
library(stringr)
library(caret)
library(PerformanceAnalytics)
library(tidyverse)


#---- Importing Loan Dataset -----

carPrice<-read.csv("CarPrice_Assignment.csv") 



#---- data Preparation -----

#cheking if there is any NA values by each column.

apply(carPrice, 2, function(x) any(is.na(x)))

# Cheking for duplicate rows

length(unique(carPrice$car_ID))== nrow(carPrice) #if true that is no duplicate valus.primary key is car_ID.


#Replacing CarName with only company name.

carPrice<- separate(carPrice,CarName,c("companyname","modelname"),sep = " ",remove = TRUE)

#eleminating model name column
carPrice<- carPrice[,-4]

#replace some mis splet column with correct spelling.
carPrice$companyname<-carPrice$companyname %>% str_replace("porcshce","porsche")
carPrice$companyname<-carPrice$companyname %>% str_replace("maxda","mazda")
carPrice$companyname<-carPrice$companyname %>% str_replace("vw","volkswagen")
carPrice$companyname<-carPrice$companyname %>% str_replace("toyouta","toyota")
carPrice$companyname<-carPrice$companyname %>% str_replace("vokswagen","volkswagen")
carPrice$companyname<-carPrice$companyname %>% str_replace("Nissan","nissan")

carPrice$companyname<- as.factor(carPrice$companyname)

#renaming column name curbweight to carweight
names(carPrice)[names(carPrice) == 'curbweight'] <- 'carweight'
str(carPrice)


# convert factors with 2 levels to numerical variables

levels(carPrice$fueltype)<- c(1,0)
carPrice$fueltype<- as.numeric(levels(carPrice$fueltype))[carPrice$fueltype]
class(carPrice$fueltype)

#creating dummy variables for all the categorical varibles to convert them into numeric one using dummyvar function.

dmy<- dummyVars("~ .",data = carPrice,fullRank = TRUE)
carPrice_dmy<- data.frame(predict(dmy,newdata = carPrice))


#creating derive metrics

#finding mean of citympg and highway mpg
carPrice_dmy$mpg<- (carPrice_dmy$citympg + carPrice_dmy$highwaympg)/2

#finding volumn for car using lenght , width and height

carPrice_dmy$carvolume<- carPrice_dmy$carlength * carPrice_dmy$carwidth * carPrice_dmy$carheight




#---- corelation matrix and scatter plot  & eda-----

cor_mat<- cor(carPrice_dmy) # The default is pearson correlation coefficient which measures the linear dependence between two variables

chart.Correlation(carPrice_dmy, histogram=TRUE, pch=19)

#frequecy distribution of enginesize,peakrpm,companyname,fuelsystem,aspiration
ggplot(carPrice,aes(x=carPrice$enginesize)) + geom_bar()
ggplot(carPrice,aes(x=carPrice$peakrpm)) + geom_bar()
ggplot(carPrice,aes(x=carPrice$companyname)) + geom_bar()
ggplot(carPrice,aes(x=carPrice$fuelsystem)) + geom_bar()
ggplot(carPrice,aes(x=carPrice$aspiration)) + geom_bar()


#plotting observed price vs engine size
#from the grpah we can say that there is a some  relationship between engine size and observed price.
ggplot(carPrice, aes(x=carPrice$enginesize, y=carPrice$price)) + geom_point()
ggplot(carPrice,aes(x=carPrice$peakrpm, y=carPrice$price)) + geom_point()
ggplot(carPrice_dmy,aes(x=carPrice_dmy$mpg, y=carPrice$price)) + geom_point()

#---- Multiple Linear Regression Model Building -----

# separate training and testing data

set.seed(100)

trainindices= sample(1:nrow(carPrice_dmy), 0.7*nrow(carPrice_dmy))
train = carPrice_dmy[trainindices,]
test = carPrice_dmy[-trainindices,]

# Build model 1 containing all variables

model_1 <-lm(price~.,data=train)
summary(model_1)
step <- stepAIC(model_1, direction="both")

step


model_2<- lm(formula = price ~ car_ID + companyname.bmw + companyname.buick + 
               companyname.chevrolet + companyname.dodge + companyname.honda + 
               companyname.isuzu + companyname.mazda + companyname.mercury + 
               companyname.mitsubishi + companyname.nissan + companyname.peugeot + 
               companyname.plymouth + companyname.porsche + companyname.renault + 
               companyname.saab + companyname.subaru + companyname.toyota + 
               companyname.volkswagen + companyname.volvo + aspiration.turbo + 
               carbody.hardtop + carbody.hatchback + carbody.sedan + carbody.wagon + 
               drivewheel.rwd + enginelocation.rear + wheelbase + carlength + 
               carwidth + carheight + carweight + enginetype.rotor + cylindernumber.five + 
               enginesize + fuelsystem.2bbl + stroke + peakrpm + carvolume, 
             data = train)


summary(model_2)
vif(model_2)

#eleminating car_ID

model_3<-lm(formula = price ~companyname.bmw + companyname.buick + 
              companyname.chevrolet + companyname.dodge + companyname.honda + 
              companyname.isuzu + companyname.mazda + companyname.mercury + 
              companyname.mitsubishi + companyname.nissan + companyname.peugeot + 
              companyname.plymouth + companyname.porsche + companyname.renault + 
              companyname.saab + companyname.subaru + companyname.toyota + 
              companyname.volkswagen + companyname.volvo + aspiration.turbo + 
              carbody.hardtop + carbody.hatchback + carbody.sedan + carbody.wagon + 
              drivewheel.rwd + enginelocation.rear + wheelbase + carlength + 
              carwidth + carheight + carweight + enginetype.rotor + cylindernumber.five + 
              enginesize + fuelsystem.2bbl + stroke + peakrpm + carvolume, 
            data = train)


summary(model_3)
vif(model_3)

#eliminating carvolume

model_4<-lm(formula = price ~companyname.bmw + companyname.buick + 
              companyname.chevrolet + companyname.dodge + companyname.honda + 
              companyname.isuzu + companyname.mazda + companyname.mercury + 
              companyname.mitsubishi + companyname.nissan + companyname.peugeot + 
              companyname.plymouth + companyname.porsche + companyname.renault + 
              companyname.saab + companyname.subaru + companyname.toyota + 
              companyname.volkswagen + companyname.volvo + aspiration.turbo + 
              carbody.hardtop + carbody.hatchback + carbody.sedan + carbody.wagon + 
              drivewheel.rwd + enginelocation.rear + wheelbase + carlength + 
              carwidth + carheight + carweight + enginetype.rotor + cylindernumber.five + 
              enginesize + fuelsystem.2bbl + stroke + peakrpm, 
            data = train)


summary(model_4)
vif(model_4)


#eleminating carlenght



model_5<-lm(formula = price ~companyname.bmw + companyname.buick + 
              companyname.chevrolet + companyname.dodge + companyname.honda + 
              companyname.isuzu + companyname.mazda + companyname.mercury + 
              companyname.mitsubishi + companyname.nissan + companyname.peugeot + 
              companyname.plymouth + companyname.porsche + companyname.renault + 
              companyname.saab + companyname.subaru + companyname.toyota + 
              companyname.volkswagen + companyname.volvo + aspiration.turbo + 
              carbody.hardtop + carbody.hatchback + carbody.sedan + carbody.wagon + 
              drivewheel.rwd + enginelocation.rear + wheelbase +carwidth + carheight + carweight + enginetype.rotor + cylindernumber.five + 
              enginesize + fuelsystem.2bbl + stroke + peakrpm, 
              data = train)


summary(model_5)
vif(model_5)

#eleminating carheight

model_6<-lm(formula = price ~companyname.bmw + companyname.buick + 
              companyname.chevrolet + companyname.dodge + companyname.honda + 
              companyname.isuzu + companyname.mazda + companyname.mercury + 
              companyname.mitsubishi + companyname.nissan + companyname.peugeot + 
              companyname.plymouth + companyname.porsche + companyname.renault + 
              companyname.saab + companyname.subaru + companyname.toyota + 
              companyname.volkswagen + companyname.volvo + aspiration.turbo + 
              carbody.hardtop + carbody.hatchback + carbody.sedan + carbody.wagon + 
              drivewheel.rwd + enginelocation.rear + wheelbase +carwidth +carweight + enginetype.rotor + cylindernumber.five + 
              enginesize + fuelsystem.2bbl + stroke + peakrpm, 
             data = train)

summary(model_6)
vif(model_6)

#eleminating carbody.hatchback

model_7<-lm(formula = price ~companyname.bmw + companyname.buick + 
              companyname.chevrolet + companyname.dodge + companyname.honda + 
              companyname.isuzu + companyname.mazda + companyname.mercury + 
              companyname.mitsubishi + companyname.nissan + companyname.peugeot + 
              companyname.plymouth + companyname.porsche + companyname.renault + 
              companyname.saab + companyname.subaru + companyname.toyota + 
              companyname.volkswagen + companyname.volvo + aspiration.turbo + 
              carbody.hardtop +carbody.sedan + carbody.wagon + 
              drivewheel.rwd + enginelocation.rear + wheelbase +carwidth +carweight + enginetype.rotor + cylindernumber.five + 
              enginesize + fuelsystem.2bbl + stroke + peakrpm, 
             data = train)

summary(model_7)
vif(model_7)

#eleminating wheelbase

model_8<-lm(formula = price ~companyname.bmw + companyname.buick + 
              companyname.chevrolet + companyname.dodge + companyname.honda + 
              companyname.isuzu + companyname.mazda + companyname.mercury + 
              companyname.mitsubishi + companyname.nissan + companyname.peugeot + 
              companyname.plymouth + companyname.porsche + companyname.renault + 
              companyname.saab + companyname.subaru + companyname.toyota + 
              companyname.volkswagen + companyname.volvo + aspiration.turbo + 
              carbody.hardtop +carbody.sedan + carbody.wagon + 
              drivewheel.rwd + enginelocation.rear +carwidth +carweight + enginetype.rotor + cylindernumber.five + 
              enginesize + fuelsystem.2bbl + stroke + peakrpm, 
            data = train)

summary(model_8)
vif(model_8)

#eleminating carwidth

model_9<-lm(formula = price ~companyname.bmw + companyname.buick + 
              companyname.chevrolet + companyname.dodge + companyname.honda + 
              companyname.isuzu + companyname.mazda + companyname.mercury + 
              companyname.mitsubishi + companyname.nissan + companyname.peugeot + 
              companyname.plymouth + companyname.porsche + companyname.renault + 
              companyname.saab + companyname.subaru + companyname.toyota + 
              companyname.volkswagen + companyname.volvo + aspiration.turbo + 
              carbody.hardtop +carbody.sedan + carbody.wagon + 
              drivewheel.rwd + enginelocation.rear + carweight + enginetype.rotor + cylindernumber.five + 
              enginesize + fuelsystem.2bbl + stroke + peakrpm, 
            data = train)

summary(model_9)
vif(model_9)

res<-cor(carPrice_dmy$enginesize,carPrice_dmy$carweight)

#eliminating carweight


model_10<-lm(formula = price ~companyname.bmw + companyname.buick + 
              companyname.chevrolet + companyname.dodge + companyname.honda + 
              companyname.isuzu + companyname.mazda + companyname.mercury + 
              companyname.mitsubishi + companyname.nissan + companyname.peugeot + 
              companyname.plymouth + companyname.porsche + companyname.renault + 
              companyname.saab + companyname.subaru + companyname.toyota + 
              companyname.volkswagen + companyname.volvo + aspiration.turbo + 
              carbody.hardtop +carbody.sedan + carbody.wagon + 
              drivewheel.rwd + enginelocation.rear + enginetype.rotor + cylindernumber.five + 
              enginesize + fuelsystem.2bbl + stroke + peakrpm, 
            data = train)

summary(model_10)
vif(model_10)

#eleminating carbody.hardtop

model_11<-lm(formula = price ~companyname.bmw + companyname.buick + 
               companyname.chevrolet + companyname.dodge + companyname.honda + 
               companyname.isuzu + companyname.mazda + companyname.mercury + 
               companyname.mitsubishi + companyname.nissan + companyname.peugeot + 
               companyname.plymouth + companyname.porsche + companyname.renault + 
               companyname.saab + companyname.subaru + companyname.toyota + 
               companyname.volkswagen + companyname.volvo + aspiration.turbo + carbody.sedan + carbody.wagon + 
               drivewheel.rwd + enginelocation.rear + enginetype.rotor + cylindernumber.five + 
               enginesize + fuelsystem.2bbl + stroke + peakrpm, 
             data = train)

summary(model_11)
vif(model_11)

#eleminating carbody.sedan

model_12<-lm(formula = price ~companyname.bmw + companyname.buick + 
               companyname.chevrolet + companyname.dodge + companyname.honda + 
               companyname.isuzu + companyname.mazda + companyname.mercury + 
               companyname.mitsubishi + companyname.nissan + companyname.peugeot + 
               companyname.plymouth + companyname.porsche + companyname.renault + 
               companyname.saab + companyname.subaru + companyname.toyota + 
               companyname.volkswagen + companyname.volvo + aspiration.turbo + carbody.wagon + 
               drivewheel.rwd + enginelocation.rear + enginetype.rotor + cylindernumber.five + 
               enginesize + fuelsystem.2bbl + stroke + peakrpm, 
             data = train)

summary(model_12)
vif(model_12)

#eleminating carbody.wagon


model_13<-lm(formula = price ~companyname.bmw + companyname.buick + 
               companyname.chevrolet + companyname.dodge + companyname.honda + 
               companyname.isuzu + companyname.mazda + companyname.mercury + 
               companyname.mitsubishi + companyname.nissan + companyname.peugeot + 
               companyname.plymouth + companyname.porsche + companyname.renault + 
               companyname.saab + companyname.subaru + companyname.toyota + 
               companyname.volkswagen + companyname.volvo + aspiration.turbo+ 
               drivewheel.rwd + enginelocation.rear + enginetype.rotor + cylindernumber.five + 
               enginesize + fuelsystem.2bbl + stroke + peakrpm, 
             data = train)

summary(model_13)
vif(model_13)

res_1<-cor(carPrice_dmy$enginesize,carPrice_dmy$peakrpm)

res_2<- cor(carPrice_dmy$enginesize,carPrice_dmy$stroke)


#eleminatin cylindernumber.five
model_14<-lm(formula = price ~companyname.bmw + companyname.buick + 
               companyname.chevrolet + companyname.dodge + companyname.honda + 
               companyname.isuzu + companyname.mazda + companyname.mercury + 
               companyname.mitsubishi + companyname.nissan + companyname.peugeot + 
               companyname.plymouth + companyname.porsche + companyname.renault + 
               companyname.saab + companyname.subaru + companyname.toyota + 
               companyname.volkswagen + companyname.volvo + aspiration.turbo+ 
               drivewheel.rwd + enginelocation.rear + enginetype.rotor+
               enginesize + fuelsystem.2bbl + stroke + peakrpm, 
             data = train)

summary(model_14)
vif(model_14)

#eleminating fuelsystem.2bbl

model_15<-lm(formula = price ~companyname.bmw + companyname.buick + 
               companyname.chevrolet + companyname.dodge + companyname.honda + 
               companyname.isuzu + companyname.mazda + companyname.mercury + 
               companyname.mitsubishi + companyname.nissan + companyname.peugeot + 
               companyname.plymouth + companyname.porsche + companyname.renault + 
               companyname.saab + companyname.subaru + companyname.toyota + 
               companyname.volkswagen + companyname.volvo + aspiration.turbo+ 
               drivewheel.rwd + enginelocation.rear + enginetype.rotor+
               enginesize + stroke + peakrpm, 
             data = train)

summary(model_15)
vif(model_15)

#eleminating commpanyname.volvo

model_16<-lm(formula = price ~companyname.bmw + companyname.buick + 
               companyname.chevrolet + companyname.dodge + companyname.honda + 
               companyname.isuzu + companyname.mazda + companyname.mercury + 
               companyname.mitsubishi + companyname.nissan + companyname.peugeot + 
               companyname.plymouth + companyname.porsche + companyname.renault + 
               companyname.saab + companyname.subaru + companyname.toyota + 
               companyname.volkswagen +aspiration.turbo+ 
               drivewheel.rwd + enginelocation.rear + enginetype.rotor+
               enginesize + stroke + peakrpm, 
             data = train)

summary(model_16)
vif(model_16)

res_3<-  cor(carPrice_dmy$enginesize,carPrice_dmy$enginetype.rotor)

##eleminating companyname.chevrolet

model_17<-lm(formula = price ~companyname.bmw + companyname.buick + 
               companyname.dodge + companyname.honda + 
               companyname.isuzu + companyname.mazda + companyname.mercury + 
               companyname.mitsubishi + companyname.nissan + companyname.peugeot + 
               companyname.plymouth + companyname.porsche + companyname.renault + 
               companyname.saab + companyname.subaru + companyname.toyota + 
               companyname.volkswagen +aspiration.turbo+ 
               drivewheel.rwd + enginelocation.rear + enginetype.rotor+
               enginesize + stroke + peakrpm, 
             data = train)

summary(model_17)
vif(model_17)

##eleminating companyname.peugeot


model_18<-lm(formula = price ~companyname.bmw + companyname.buick + 
               companyname.dodge + companyname.honda + 
               companyname.isuzu + companyname.mazda + companyname.mercury + 
               companyname.mitsubishi + companyname.nissan + 
               companyname.plymouth + companyname.porsche + companyname.renault + 
               companyname.saab + companyname.subaru + companyname.toyota + 
               companyname.volkswagen +aspiration.turbo+ 
               drivewheel.rwd + enginelocation.rear + enginetype.rotor+
               enginesize + stroke + peakrpm, 
             data = train)

summary(model_18)
vif(model_18)

##eleminating companyname.mercury


model_19<-lm(formula = price ~companyname.bmw + companyname.buick + 
               companyname.dodge + companyname.honda + 
               companyname.isuzu + companyname.mazda + 
               companyname.mitsubishi + companyname.nissan + 
               companyname.plymouth + companyname.porsche + companyname.renault + 
               companyname.saab + companyname.subaru + companyname.toyota + 
               companyname.volkswagen +aspiration.turbo+ 
               drivewheel.rwd + enginelocation.rear + enginetype.rotor+
               enginesize + stroke + peakrpm, 
             data = train)

summary(model_19)
vif(model_19)

##eleminating companyname.saab

model_20<-lm(formula = price ~companyname.bmw + companyname.buick + 
               companyname.dodge + companyname.honda + 
               companyname.isuzu + companyname.mazda + 
               companyname.mitsubishi + companyname.nissan + 
               companyname.plymouth + companyname.porsche + companyname.renault + 
               companyname.subaru + companyname.toyota + 
               companyname.volkswagen +aspiration.turbo+ 
               drivewheel.rwd + enginelocation.rear + enginetype.rotor+
               enginesize + stroke + peakrpm, 
             data = train)

summary(model_20)
vif(model_20)


##eleminating enginelocation.rear

model_21<-lm(formula = price ~companyname.bmw + companyname.buick + 
               companyname.dodge + companyname.honda + 
               companyname.isuzu + companyname.mazda + 
               companyname.mitsubishi + companyname.nissan + 
               companyname.plymouth + companyname.porsche + companyname.renault + 
               companyname.subaru + companyname.toyota + 
               companyname.volkswagen +aspiration.turbo+ 
               drivewheel.rwd +enginetype.rotor+
               enginesize + stroke + peakrpm, 
             data = train)

summary(model_21)
vif(model_21)

###eleminating stroke

model_22<-lm(formula = price ~companyname.bmw + companyname.buick + 
               companyname.dodge + companyname.honda + 
               companyname.isuzu + companyname.mazda + 
               companyname.mitsubishi + companyname.nissan + 
               companyname.plymouth + companyname.porsche + companyname.renault + 
               companyname.subaru + companyname.toyota + 
               companyname.volkswagen +aspiration.turbo+ 
               drivewheel.rwd +enginetype.rotor+ enginesize + peakrpm, 
               data = train)

summary(model_22)
vif(model_22)

##eleminating companyname.isuzu

model_23<-lm(formula = price ~companyname.bmw + companyname.buick + 
               companyname.dodge + companyname.honda + companyname.mazda + 
               companyname.mitsubishi + companyname.nissan + 
               companyname.plymouth + companyname.porsche + companyname.renault + 
               companyname.subaru + companyname.toyota + 
               companyname.volkswagen +aspiration.turbo+ 
               drivewheel.rwd +enginetype.rotor+ enginesize + peakrpm, 
             data = train)

summary(model_23)
vif(model_23)

res_4<-  cor(carPrice_dmy$enginesize,carPrice_dmy$drivewheel.rwd)

##eleminating drivewheel.rwd

model_24<-lm(formula = price ~companyname.bmw + companyname.buick + 
               companyname.dodge + companyname.honda + companyname.mazda + 
               companyname.mitsubishi + companyname.nissan + 
               companyname.plymouth + companyname.porsche + companyname.renault + 
               companyname.subaru + companyname.toyota + 
               companyname.volkswagen +aspiration.turbo+enginetype.rotor+ enginesize + peakrpm, 
             data = train)

summary(model_24)
vif(model_24)


#eleminating companyname.volkswagen
model_25<-lm(formula = price ~companyname.bmw + companyname.buick + 
               companyname.dodge + companyname.honda + companyname.mazda + 
               companyname.mitsubishi + companyname.nissan + 
               companyname.plymouth + companyname.porsche + companyname.renault + 
               companyname.subaru + companyname.toyota + 
               aspiration.turbo+enginetype.rotor+ enginesize + peakrpm, 
             data = train)

summary(model_25)
vif(model_25)


#eleminating companyname.subaru
model_26<-lm(formula = price ~companyname.bmw + companyname.buick + 
               companyname.dodge + companyname.honda + companyname.mazda + 
               companyname.mitsubishi + companyname.nissan + 
               companyname.plymouth + companyname.porsche + companyname.renault + 
               companyname.toyota + 
               aspiration.turbo+enginetype.rotor+ enginesize + peakrpm, 
             data = train)

summary(model_26)
vif(model_26)

#eleminating companyname.mazda
model_27<-lm(formula = price ~companyname.bmw + companyname.buick + 
               companyname.dodge + companyname.honda + 
               companyname.mitsubishi + companyname.nissan + 
               companyname.plymouth + companyname.porsche + companyname.renault + 
               companyname.toyota + 
               aspiration.turbo+enginetype.rotor+ enginesize + peakrpm, 
             data = train)

summary(model_27)
vif(model_27)

##eleminating companyname.renault
model_28<-lm(formula = price ~companyname.bmw + companyname.buick + 
               companyname.dodge + companyname.honda + 
               companyname.mitsubishi + companyname.nissan + 
               companyname.plymouth + companyname.porsche +companyname.toyota + 
               aspiration.turbo+enginetype.rotor+ enginesize + peakrpm, 
             data = train)

summary(model_28)
vif(model_28)

##eleminating companyname.honda

model_29<-lm(formula = price ~companyname.bmw + companyname.buick + 
               companyname.dodge +companyname.mitsubishi + companyname.nissan + 
               companyname.plymouth + companyname.porsche +companyname.toyota + 
               aspiration.turbo+enginetype.rotor+ enginesize + peakrpm, 
             data = train)

summary(model_29)
vif(model_29)

##eleminating companyname.toyota

model_30<-lm(formula = price ~companyname.bmw + companyname.buick + 
               companyname.dodge +companyname.mitsubishi + companyname.nissan + 
               companyname.plymouth + companyname.porsche+ 
               aspiration.turbo+enginetype.rotor+ enginesize + peakrpm, 
             data = train)

summary(model_30)
vif(model_30)

#eleminating companyname.nissan

model_31<-lm(formula = price ~companyname.bmw + companyname.buick + 
               companyname.dodge +companyname.mitsubishi+ 
               companyname.plymouth + companyname.porsche+ 
               aspiration.turbo+enginetype.rotor+ enginesize + peakrpm, 
             data = train)

summary(model_31)
vif(model_31)

#eleminating companyname.plymouth

model_32<-lm(formula = price ~companyname.bmw + companyname.buick + 
               companyname.dodge +companyname.mitsubishi+ companyname.porsche+ 
               aspiration.turbo+enginetype.rotor+ enginesize + peakrpm, 
             data = train)

summary(model_32)
vif(model_32)

#eleminating companyname.dodge
model_33<-lm(formula = price ~companyname.bmw + companyname.buick + 
               companyname.mitsubishi+ companyname.porsche+ 
               aspiration.turbo+enginetype.rotor+ enginesize + peakrpm, 
             data = train)

summary(model_33)
vif(model_33)

#eleminating companyname.mitsubishi
model_34<-lm(formula = price ~companyname.bmw + companyname.buick + 
               companyname.porsche+aspiration.turbo+enginetype.rotor+ 
               enginesize + peakrpm,data = train)

summary(model_34)
vif(model_34)


#model_34 is my final model all of the varibles  has p value less than p<0.05 with a three star. and vif is also under 2.
#Multiple R-squared:  0.9139,	Adjusted R-squared:  0.9094 . So this two parameter also very close to each other
#below i have plotted the error vector and from that plot we can see that error is very random and we can't preddicts any pattern from that.



#----- predicting the results in test dataset and calculating error----

Predict_1 <- predict(model_34,test[,-66])
test$predicted_price<- Predict_1

r <- cor(test$price,test$predicted_price)
rsquared <- cor(test$price,test$predicted_price)^2

rsquared

# Function that returns Root Mean Squared Error
rmse <- function(error)
{
  sqrt(mean(error^2))
}

# Function that returns Mean Absolute Error
mae <- function(error)
{
  mean(abs(error))
}

test$error<- (test$price -test$predicted_price)

rmse(test$error)
mae(test$error)

#errors are randomly distributed from the graph we can conclue that.

ggplot(test, aes(x=test$car_ID, y=test$error)) + geom_point(size=2)



#plot of actual price vs predicted price
plot(test$car_ID,test$price,type="l",col="red")
lines(test$car_ID,test$predicted_price,col="green")









#-----Model Explanation----

#So in the above model_34 we have seven variables in total out of that three of
#them is the BRAND name.So from that we can colclude there is a brand value tagged in every car pricing.
# apart from that we can say that RPM, Turbo, EngineSize are also driving factor for a pricing of
# a car.


