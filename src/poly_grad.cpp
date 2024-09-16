// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(StanHeaders)]]
#include <stan/math/mix.hpp> // stuff from mix/ must come first
#include <stan/math.hpp>     // finally pull in everything from rev/ and prim/
#include <Rcpp.h>
#include <RcppEigen.h>       // do this AFTER including stuff from stan/math
//using namespace stan;
//using namespace stan::math;

// [[Rcpp::plugins(cpp17)]]

// [[Rcpp::export]]
Eigen::MatrixXd H(Eigen::VectorXd x, Eigen::VectorXd a) { // Hessian by AD using Stan
  double fx;
  Eigen::VectorXd grad_fx;
  Eigen::MatrixXd H;
  using stan::math::dot_self;
  stan::math::hessian([&a](auto x) { return dot_self(x - a); },
                      x, fx, grad_fx, H);
  return H;
}



Eigen::Matrix<stan::math::var_value<double>, -1, 1> log_cholesky_grad(Eigen::Matrix<stan::math::var_value<double>, Eigen::Dynamic, 1> l_params, const Eigen::MatrixXd& A) {
  int n = A.rows(); // assuming A is square

  // Function to calculate Tr(LL^T A) with log-Cholesky parameterization
  auto func = [&](auto &l_params) -> auto {
    // Use Eigen matrix with stan::math::var type to support autodiff
    //Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> L(n, n);
    Eigen::Matrix<stan::math::fvar<stan::math::var_value<double> >, -1, -1> L(n,n);
    
    L.setZero();

    // Fill the lower triangular matrix L with the log-Cholesky parameterization
    int index = 0;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j <= i; ++j) {
        if (i == j) {
          L(i, j) =  stan::math::exp(l_params(index,0)); // exponentiate diagonal elements
        } else {
          L(i, j) = l_params(index,0); // off-diagonal elements remain the same
        }
        index++;
      }
    }

    // Compute the trace of L * L^T * A
    Eigen::Matrix<stan::math::fvar<stan::math::var_value<double> >, Eigen::Dynamic, Eigen::Dynamic> LLT = L * L.transpose();
    stan::math::fvar<stan::math::var_value<double> > result = (LLT.cwiseProduct(A)).sum(); // Tr(LL^T A)
    return result;
  };

  // Prepare storage for gradient
  stan::math::var_value<double> fx;
  Eigen::Matrix<stan::math::var_value<double> , -1, 1> grad_fx;

  // Compute gradient
  stan::math::gradient(func, l_params, fx, grad_fx);

  return grad_fx; // Return the gradient with respect to log-Cholesky parameters
}




Eigen::Matrix< stan::math::fvar<stan::math::fvar<stan::math::fvar<stan::math::var_value<double> >>>, -1,-1 > log_cholesky_H(Eigen::Matrix<stan::math::fvar<stan::math::fvar<stan::math::var_value<double> >>, Eigen::Dynamic, 1> l_params, const Eigen::MatrixXd& A) {
  int n = A.rows(); // assuming A is square

  // Function to calculate Tr(LL^T A) with log-Cholesky parameterization
  auto func = [&](auto &l_params) -> auto {
    // Use Eigen matrix with stan::math::var type to support autodiff
    //Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> L(n, n);
    Eigen::Matrix<stan::math::fvar<stan::math::fvar<stan::math::fvar<stan::math::fvar<stan::math::var_value<double> >>>>, -1, -1> L(n,n);
    
    L.setZero();

    // Fill the lower triangular matrix L with the log-Cholesky parameterization
    int index = 0;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j <= i; ++j) {
        if (i == j) {
          L(i, j) =  stan::math::exp(l_params(index,0)); // exponentiate diagonal elements
        } else {
          L(i, j) = l_params(index,0); // off-diagonal elements remain the same
        }
        index++;
      }
    }

    // Compute the trace of L * L^T * A
    Eigen::Matrix<stan::math::fvar<stan::math::fvar<stan::math::var_value<double> > >, Eigen::Dynamic, Eigen::Dynamic> LLT = L * L.transpose();
    stan::math::fvar<stan::math::fvar<stan::math::var_value<double> >> result = (LLT.cwiseProduct(A)).sum(); // Tr(LL^T A)
    return result;
  };

  // Prepare storage for gradient
  stan::math::fvar<stan::math::fvar<stan::math::var_value<double> >> fx;
  Eigen::Matrix<stan::math::fvar<stan::math::fvar<stan::math::var_value<double> > >, -1, 1> grad_fx;
  Eigen::Matrix< stan::math::fvar<stan::math::fvar<stan::math::var_value<double> > >, Eigen::Dynamic, Eigen::Dynamic >  hess_fx;

  // Compute gradient
  stan::math::hessian(func, l_params, fx, grad_fx, hess_fx);

  return hess_fx; // Return the gradient with respect to log-Cholesky parameters
}

