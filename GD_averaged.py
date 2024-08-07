import numpy as np
import sys
import torch
from torch import nn



class Network(nn.Module):
    def __init__(self, D, M, sigma, diagonal=False):
        super(Network, self).__init__()
        self.D = D
        self.M = M
        self.fc1 = nn.Linear(D, M, bias=False)
        self.fc2 = torch.ones((M, 1), requires_grad=False) / np.sqrt(self.M)
        self.sigma = sigma

        if diagonal==False:
            # Set the weights as gaussian random variables
            self.fc1.weight.data = torch.normal(0, 1, (M, D))
        else:
            # Set the weights as a diagonal matrix
            self.fc1.weight.data = torch.eye(M,D)*np.sqrt(D)

    def forward(self, x):
        x = self.fc1(x) / np.sqrt(self.D) + torch.normal(0, self.sigma, (x.shape[0], self.M), requires_grad=False)
        x = (x**2) @ self.fc2
        return x


def main(D, alpha, beta, beta_star, sigma, reg, lr, T, samples_average, samples):
    # teacher_type = "diagonal"
    teacher_type = "gaussian"


    M = int(D * beta)
    M_star = int(D * beta_star)
    N = int(D**2 * alpha)

    gen_error = np.ones((samples))
    S_all_final = np.ones((samples, samples_average, D, D))
    for s_global in range(samples):
        with torch.no_grad():
            teacher = Network(D, M_star, sigma, diagonal=(teacher_type=="diagonal"))
            X = torch.randn(N, D, requires_grad=False)
            y = teacher(X)


        S_final = np.ones((samples_average, D, D))
        for s in range(samples_average):
            # Initialise the teacher and student networks
            student = Network(D, M, 0.0)


            # Optimizer
            optimizer = torch.optim.SGD(student.parameters(), lr=lr, weight_decay=reg)

            # The training loop
            for t in range(T):
                # Compute the gradient of the loss with respect to the student network parameters
                y_pred = student(X)
                # Add a sigma*sqrt(d) deterministic shift to the output
                y_pred = y_pred 

                loss = ((y_pred - y)**2).sum()/4
                loss.backward()

                # Update the student network parameters
                optimizer.step()
                optimizer.zero_grad()

            # Compute the generalization error
            with torch.no_grad():
                W = student.fc1.weight.data.numpy()
                S_final[s] = W.T @ W / M

        S_all_final[s_global] = S_final
                
        # Compute the generalization error
        with torch.no_grad():
            W_star = teacher.fc1.weight.data.numpy()
            S_star = W_star.T @ W_star / M_star
            
            gen_error[s_global] = np.sum((S_final.mean(axis=0) - S_star)**2)/D

    # Save the results
    np.save(f"data_simulations_averaged_noise_teacher_reg/gen_error_{D}_{alpha}_{beta}_{beta_star}_{sigma}_{reg}_{samples_average}_{lr}_{teacher_type}.npy", gen_error)
    np.save(f"data_simulations_averaged_noise_teacher_reg/S_all_final_{D}_{alpha}_{beta}_{beta_star}_{sigma}_{reg}_{samples_average}_{lr}_{teacher_type}.npy", S_all_final)


if __name__ == '__main__':
    # Get the parameters from the command line
    print(sys.argv)
    D = int(sys.argv[1])
    alpha = float(sys.argv[2])
    beta = float(sys.argv[3])
    beta_star = float(sys.argv[4])
    sigma = float(sys.argv[5])
    reg = float(sys.argv[6])
    samples_average = int(sys.argv[7])
    
    lr = 0.1
    T = 1000
    samples = 2

    main(D, alpha, beta, beta_star, sigma, reg, lr, T, samples_average, samples)