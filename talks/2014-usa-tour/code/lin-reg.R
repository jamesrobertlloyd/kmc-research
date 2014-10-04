# Create data with outliers

N <- 50

x <- seq(0, 1, length=N)
y <- x

y[1] <- +50
y[N] <- -50

# Plot data

plot(x, y, main='Scatter plot')

# Plot without the outliers

plot(x[2:(N-1)], y[2:(N-1)], main='Without outliers')

# Fit and summarise linear model

my.lm <- lm(y ~ x)

print(summary(my.lm))