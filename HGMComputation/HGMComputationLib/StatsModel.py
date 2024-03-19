import HGMComputationLib.manifolds as manifolds
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
import numpy as np


def FrechetMean_Kendall3D(dataList, maxIter=5000, tol=0.001, stepsize=1):
    mu = dataList[0]
    nManDim = dataList[0].nPt
    nData = len(dataList)

    for i in range(maxIter):
        dMu = manifolds.kendall3D_tVec(nManDim)

        for j in range(nData):
            Log_mu_to_y_j = mu.LogMap(dataList[j])

            for d in range(nManDim):
                for k in range(3):
                    dMu.tVector[k][d] += stepsize * ((1.0 / nData) * Log_mu_to_y_j.tVector[k][d])

        Mu_i = mu.ExponentialMap(dMu)
        print("Iteration " + str(i) + " Step size: ", mu.LogMap(Mu_i).norm())
        if mu.LogMap(Mu_i).norm() < tol:
            return Mu_i
        else:
            mu = Mu_i
    print("Frechet mean didn't converge.")
    return mu


def LinearizedGeodesicRegression_Kendall3D(t_list, pt_list, max_iter=100, stepSize=0.05, step_tol=1e-8,
                                           useFrechetMeanAnchor=False, verbose=True):
    nData = len(pt_list)

    if verbose:
        print("=================================================================")
        print("      Linear Regression on Anchor Point Tangent Vector Space    ")
        print("=================================================================")

    # Initialize an anchor point
    if useFrechetMeanAnchor:
        p_anchor = FrechetMean_Kendall3D(pt_list)
    else:
        t_min_idx = np.argmin(t_list)

        p_anchor = pt_list[t_min_idx]

    nManifoldDim = p_anchor.nPt

    # Initial point on manifold and tangent vector
    init_Interp = manifolds.kendall3D(nManifoldDim)
    init_tVec = manifolds.kendall3D_tVec(nManifoldDim)

    base = init_Interp
    tangent = init_tVec

    # Iteration Parameters
    prevEnergy = 1e10
    prevBase = base
    prevTangent = tangent

    for i in range(max_iter):
        tVec_list = []
        w_list = []

        for d in range(nManifoldDim):
            for k in range(3):
                w_list.append([])

        for j in range(nData):
            tVec_j = p_anchor.LogMap(pt_list[j])

            for k in range(3):
                for d in range(nManifoldDim):
                    w_list[k * nManifoldDim + d].append(tVec_j.tVector[k, d])

        estModel_list = []

        for k in range(3):
            for d in range(nManifoldDim):
                t_list_sm = sm.add_constant(t_list)
                w_d_np = np.asarray(w_list[k * nManifoldDim + d])
                LS_model_d = sm.OLS(w_d_np, t_list_sm)
                est_d = LS_model_d.fit(method='qr')

                # est_d = LS_model_d.fit()
                estModel_list.append(est_d)

        # if verbose:
        # 	print( est_d.summary() )

        v_tangent_on_p_anchor = manifolds.kendall3D_tVec(nManifoldDim)
        v_to_base_on_p_anchor = manifolds.kendall3D_tVec(nManifoldDim)

        for d in range(nManifoldDim):
            v_to_base_on_p_anchor.tVector[0, d] = estModel_list[d].params[0]
            v_to_base_on_p_anchor.tVector[1, d] = estModel_list[nManifoldDim + d].params[0]
            v_to_base_on_p_anchor.tVector[2, d] = estModel_list[2 * nManifoldDim + d].params[0]

            if len(estModel_list[d].params) < 2:
                v_tangent_on_p_anchor.tVector[0, d] = 0
            else:
                v_tangent_on_p_anchor.tVector[0, d] = estModel_list[d].params[1]

            if len(estModel_list[nManifoldDim + d].params) < 2:
                v_tangent_on_p_anchor.tVector[1, d] = 0
            else:
                v_tangent_on_p_anchor.tVector[1, d] = estModel_list[nManifoldDim + d].params[1]

            if len(estModel_list[2 * nManifoldDim + d].params) < 2:
                v_tangent_on_p_anchor.tVector[2, d] = 0
            else:
                v_tangent_on_p_anchor.tVector[2, d] = estModel_list[2 * nManifoldDim + d].params[1]

        print( "Anchor point to base" )
        # print( v_to_base_on_p_anchor.tVector )
        print("Step norm: " + str(v_to_base_on_p_anchor.norm()))

        newBase = p_anchor.ExponentialMap(v_to_base_on_p_anchor)
        newTangent = p_anchor.ParallelTranslateAtoB(p_anchor, newBase, v_tangent_on_p_anchor)

        energy = 0

        for n in range(nData):
            time_pt = t_list[n]
            target = pt_list[n]

            current_tangent = manifolds.kendall3D_tVec(nManifoldDim)

            for k in range(3):
                for d in range(nManifoldDim):
                    current_tangent.tVector[k, d] = newTangent.tVector[k, d] * time_pt

            estimate_n = newBase.ExponentialMap(current_tangent)
            et = estimate_n.LogMap(target)

            # Energy of the tangential error
            energy += et.normSquared()

        if energy >= prevEnergy:
            if verbose:
                print("=========================")
                print("   Energy Increased ")
                print(energy)
                print("=========================")

            break
        else:
            prevBase = newBase
            prevTangent = newTangent
            p_anchor = newBase
            base = newBase
            tangent = newTangent
            prevEnergy = energy
            if verbose:
                print("==================================")
                print(str(i) + "th Iteration energy")
                print(energy)
                print("==================================")
        if stepSize < step_tol:
            if verbose:
                print("==================================")
                print("Step size under tolerance")
                print("Aborting")
                print("==================================")
            break

    return base, tangent


def LinearizedGeodesicPolynomialRegression_Kendall3D(t_list, pt_list, order=1, max_iter=100, stepSize=0.05, step_tol=1e-8,
                                           useFrechetMeanAnchor=False, verbose=True):
    # t_list: list of timestamps
    # pt_list: list of points in kendall (pre)space

    nData = len(pt_list)
    if verbose:
        print("=================================================================")
        print("   Polynomial Regression on Anchor Point Tangent Vector Space    ")
        print("=================================================================")

    # Initialize an anchor point
    if useFrechetMeanAnchor:
        p_anchor = FrechetMean_Kendall3D(pt_list)
    else:
        t_min_idx = np.argmin(t_list)

        p_anchor = pt_list[t_min_idx]

    nManifoldDim = p_anchor.nPt

    # Initial point on manifold and tangent vector
    init_Interp = manifolds.kendall3D(nManifoldDim)
    init_tVecs = []
    for o in range(order):
        init_tVecs.append(manifolds.kendall3D_tVec(nManifoldDim))

    base = init_Interp
    tangents = init_tVecs

    # Iteration Parameters
    prevEnergy = 1e10
    prevBase = base
    prevTangents = tangents

    for i in range(max_iter):
        tVec_list = []
        w_list = []

        for d in range(nManifoldDim):
            for k in range(3):
                w_list.append([])

        for j in range(nData):
            tVec_j = p_anchor.LogMap(pt_list[j])

            for k in range(3):
                for d in range(nManifoldDim):
                    w_list[k * nManifoldDim + d].append(tVec_j.tVector[k, d])

        estModel_list = []

        for k in range(3):
            for d in range(nManifoldDim):
                w_d_np = np.asarray(w_list[k * nManifoldDim + d])
                polynomial_features = PolynomialFeatures(degree=order)
                xp = polynomial_features.fit_transform(t_list[:, np.newaxis])
                LS_model_d = sm.OLS(w_d_np, xp)
                est_d = LS_model_d.fit(method='qr')

                # est_d = LS_model_d.fit()
                estModel_list.append(est_d)

                # if verbose:
                #     print( est_d.summary() )

        v_to_base_on_p_anchor = manifolds.kendall3D_tVec(nManifoldDim)
        v_tangent_on_p_anchors = []
        for o in range(order):
            v_tangent_on_p_anchors.append(manifolds.kendall3D_tVec(nManifoldDim))

        for k in range(3):
            for d in range(nManifoldDim):
                params = estModel_list[nManifoldDim*k+d].params
                v_to_base_on_p_anchor.tVector[k, d] = params[0]
                for o in range(order):
                    v_tangent_on_p_anchors[o].tVector[k, d] = params[o+1]

        print( "Anchor point to base" )
        # print( v_to_base_on_p_anchor.tVector )
        print("Step norm: " + str(v_to_base_on_p_anchor.norm()))

        newBase = p_anchor.ExponentialMap(v_to_base_on_p_anchor)
        newTangents = []
        for o in range(order):
            newTangents.append(p_anchor.ParallelTranslateAtoB(p_anchor, newBase, v_tangent_on_p_anchors[o]))

        energy = 0

        for n in range(nData):
            time_pt = t_list[n]
            target = pt_list[n]

            current_tangent = manifolds.kendall3D_tVec(nManifoldDim)

            for k in range(3):
                for d in range(nManifoldDim):
                    for o in range(order):
                        current_tangent.tVector[k, d] += newTangents[o].tVector[k, d] * time_pt ** (o + 1)

            estimate_n = newBase.ExponentialMap(current_tangent)
            et = estimate_n.LogMap(target)

            # Energy of the tangential error
            energy += et.normSquared()

        if energy >= prevEnergy:
            if verbose:
                print("=========================")
                print("   Energy Increased ")
                print(energy)
                print("=========================")

            break
        else:
            prevBase = newBase
            prevTangents = newTangents
            p_anchor = newBase
            base = newBase
            tangents = newTangents
            prevEnergy = energy
            if verbose:
                print("==================================")
                print(str(i) + "th Iteration energy")
                print(energy)
                print("==================================")
        if stepSize < step_tol:
            if verbose:
                print("==================================")
                print("Step size under tolerance")
                print("Aborting")
                print("==================================")
            break

    return base, tangents

def MultivariateLinearizedGeodesicRegression_Intercept_Kendall3D(X, Y, max_iter=100, stepSize=0.05, step_tol=1e-8,
                                                                 useFrechetMeanAnchor=False, verbose=True):
    # X: list of covariates
    # Y: list of p0/anchor points from level1 regression
    if verbose:
        print("========================================================================")
        print("    Multivariate Linear Regression on Cross Subjects Intercept Points   ")
        print("========================================================================")

    # 	print( "No. Independent Varibles : " + str( len( X[ 0 ] ) ) )
    # 	print( "No. Observations : " + str( len( X ) ) )

    nData = len(Y)
    nParam = len(X[0])

    # Anchor point is chosen by the last entry of covariates
    # Continuous variable such as a genetic disease score should be the last entry of covariates
    # If data don't have a continuous covariates, the last entry can be a categorical covariate
    t_list = []

    for i in range(len(X)):
        t_list.append(X[i][-1])

    # Set an anchor point
    t_min_idx = np.argmin(t_list)
    p_anchor = Y[t_min_idx]

    nManifoldDim = p_anchor.nPt

    # Initial point on manifold and tangent vector
    init_Interp = manifolds.kendall3D(nManifoldDim)
    init_tVecs = []
    for p in range(nParam):
        init_tVecs.append(manifolds.kendall3D_tVec(nManifoldDim))

    base = init_Interp
    tangents = init_tVecs

    # Iteration Parameters
    prevEnergy = 1e10
    prevBase = base
    prevTangents = tangents

    for i in range(max_iter):
        tVec_list = []
        w_list = []

        for d in range(nManifoldDim):
            for k in range(3):
                w_list.append([])

        for j in range(nData):
            tVec_j = p_anchor.LogMap(Y[j])

            for k in range(3):
                for d in range(nManifoldDim):
                    w_list[k * nManifoldDim + d].append(tVec_j.tVector[k, d])

        estModel_list = []

        for k in range(3):
            for d in range(nManifoldDim):
                t_list_sm = sm.add_constant(X)
                w_d_np = np.asarray(w_list[k * nManifoldDim + d])
                LS_model_d = sm.OLS(w_d_np, t_list_sm)
                est_d = LS_model_d.fit(method='qr')

                # est_d = LS_model_d.fit()
                estModel_list.append(est_d)

                # if verbose:
                # 	print( est_d.summary() )

        v_to_base_on_p_anchor = manifolds.kendall3D_tVec(nManifoldDim)
        v_tangent_on_p_anchors = []
        for p in range(nParam):
            v_tangent_on_p_anchors.append(manifolds.kendall3D_tVec(nManifoldDim))

        for d in range(nManifoldDim):
            for k in range(3):
                params = estModel_list[nManifoldDim * k + d].params
                v_to_base_on_p_anchor.tVector[k, d] = params[0]
                for p in range(nParam):
                    v_tangent_on_p_anchors[p].tVector[k, d] = params[p+1]

        print( "Anchor point to base" )
        # print( v_to_base_on_p_anchor.tVector )
        print("Step norm: " + str(v_to_base_on_p_anchor.norm()))

        newBase = p_anchor.ExponentialMap(v_to_base_on_p_anchor)
        newTangents = []
        for p in range(nParam):
            newTangents.append(p_anchor.ParallelTranslateAtoB(p_anchor, newBase, v_tangent_on_p_anchors[p]))

        energy = 0

        for n in range(nData):
            X_pt = X[n]
            target = Y[n]

            current_tangent = manifolds.kendall3D_tVec(nManifoldDim)

            for k in range(3):
                for d in range(nManifoldDim):
                    for p in range(nParam):
                        current_tangent.tVector[k, d] += newTangents[p].tVector[k, d] * X_pt[p]

            estimate_n = newBase.ExponentialMap(current_tangent)
            et = estimate_n.LogMap(target)

            # Energy of the tangential error
            energy += et.normSquared()

        if energy >= prevEnergy:
            if verbose:
                print("=========================")
                print("   Energy Increased ")
                print(energy)
                print("=========================")

            break
        else:
            prevBase = newBase
            prevTangents = newTangents
            p_anchor = newBase
            base = newBase
            tangents = newTangents
            prevEnergy = energy
            if verbose:
                print("==================================")
                print(str(i) + "th Iteration energy")
                print(energy)
                print("==================================")
        if stepSize < step_tol:
            if verbose:
                print("==================================")
                print("Step size under tolerance")
                print("Aborting")
                print("==================================")
            break

    return base, tangents

def MultivariateLinearizedGeodesicRegression_Slope_Kendall3D(X, Y, beta0, p0_list, tVec_intercept_arr,
                                                             cov_intercept_list, verbose=True):
    # X: list of covariates for slope regression (same as cov_intercept_list for most cases)
    # Y: list of slopes from level 1 regression

    if verbose:
        print("=============================================================")
        print("  Multivariate Linear Regression on Cross Subjects Slopes    ")
        print("=============================================================")

    if len(X) == 0 or len(X[0]) == 0:
        print("Covariates cannot be zero-length!")
        return


    nData = len(Y)
    polynomial_order = len(Y[0])
    nParam = len(X[0])

    p_anchor = beta0
    nManifoldDim = p_anchor.nPt

    w_list = []
    for o in range(polynomial_order):
        w_list.append([])
        for k in range(3):
            for d in range(nManifoldDim):
                w_list[o].append([])

    for j in range(nData):
        # Parallel translate a group-wise tangent vector to population-level intercept
        beta_tVec_f_i = manifolds.kendall3D_tVec(nManifoldDim)

        for tt in range(len(cov_intercept_list[j])):
            est_beta_tt = tVec_intercept_arr[tt]

            for kk in range(3):
                for dd in range(nManifoldDim):
                    beta_tVec_f_i.tVector[kk, dd] += (est_beta_tt.tVector[kk, dd] * cov_intercept_list[j][tt])

        tVec_j = []
        for Y_j in Y[j]:
            f_j = beta0.ExponentialMap(beta_tVec_f_i)
            Y_j_at_f_j = p0_list[j].ParallelTranslateAtoB(p0_list[j], f_j, Y_j)
            Y_j_tilde = f_j.ParallelTranslateAtoB(f_j, beta0, Y_j_at_f_j)

            tVec_j.append(Y_j_tilde)

        for o in range(polynomial_order):
            for k in range(3):
                for d in range(nManifoldDim):
                    w_list[o][k * nManifoldDim + d].append(tVec_j[o].tVector[k, d])

    estModel_list = []
    for o in range(polynomial_order):
        estModel_list.append([])
        for k in range(3):
            for d in range(nManifoldDim):
                X_sm = sm.add_constant(X)
                w_d_np = np.asarray(w_list[o][k * nManifoldDim + d])
                LS_model_d = sm.OLS(w_d_np, X_sm)

                est_d = LS_model_d.fit()
                estModel_list[o].append(est_d)

    # if verbose:
    # 	print( est_d.summary() )

    # base slope for t
    v_t_list = []
    for o in range(polynomial_order):
        v_t = manifolds.kendall3D_tVec(nManifoldDim)
        for k in range(3):
            for d in range(nManifoldDim):
                v_t.tVector[k, d] = estModel_list[o][k * nManifoldDim + d].params[0]
        v_t_list.append(v_t)

    new_tVec_arr = []
    for o in range(polynomial_order):
        new_tVec_arr.append([])
        for par in range(nParam):
            v_tangent_on_p_anchor_param = manifolds.kendall3D_tVec(nManifoldDim)

            for k in range(3):
                for d in range(nManifoldDim):
                    v_tangent_on_p_anchor_param.tVector[k, d] = estModel_list[o][k * nManifoldDim + d].params[par + 1]

            new_tVec_arr[o].append(v_tangent_on_p_anchor_param)

    # Append time-wise slope tangent vector at the last
    for o in range(polynomial_order):
        new_tVec_arr[o].append(v_t_list[o])
    tangent_arr = new_tVec_arr

    return tangent_arr


def MultivariateLinearizedGeodesicPolynomialRegression_Intercept_Kendall3D(X, Y, max_iter=100, order=1, stepSize=0.05,
                                                                           step_tol=1e-8, useFrechetMeanAnchor=False,
                                                                           verbose=True):
    # X: list of covariates
    # Y: list of p0/anchor points from level1 regression
    if verbose:
        print("===========================================================================")
        print("    Multivariate Polynomial Regression on Cross Subjects Intercept Points  ")
        print("===========================================================================")

    # 	print( "No. Independent Varibles : " + str( len( X[ 0 ] ) ) )
    # 	print( "No. Observations : " + str( len( X ) ) )

    nData = len(Y)
    nParam = len(X[0])
    print(X[0])
    
    print(nData)
    print(nParam)

    # Anchor point is chosen by the last entry of covariates
    # Continuous variable such as a genetic disease score should be the last entry of covariates
    # If data don't have a continuous covariates, the last entry can be a categorical covariate
    t_list = []

    for i in range(len(X)):
        t_list.append(X[i][-1])

    # Set an anchor point
    t_min_idx = np.argmin(t_list)
    p_anchor = Y[t_min_idx]

    # elevate covariates to higher order if needed
    new_X = []
    new_nParam = nParam
    if order == 1:
        for i in range(nData):
            new_X.append(X[i].copy())
    elif order == 2:
        # quadratic expansion
        new_nParam = int((nParam ** 2 + 3 * nParam) / 2)
        for i in range(nData):
            new_x = X[i].copy()
            for j in range(nParam):
                new_x.append(X[i][j] ** 2)
            for j in range(nParam - 1):
                for k in range(j+1, nParam):
                    new_x.append(X[i][j] * X[i][k])
            new_X.append(new_x)
    else:
        raise Exception("The polynomial order of multivariate regression has not been implemented.")

    nManifoldDim = p_anchor.nPt

    # Initial point on manifold and tangent vector
    init_Interp = manifolds.kendall3D(nManifoldDim)
    init_tVecs = []
    for p in range(new_nParam):
        init_tVecs.append(manifolds.kendall3D_tVec(nManifoldDim))

    base = init_Interp
    tangents = init_tVecs

    # Iteration Parameters
    prevEnergy = 1e10
    prevBase = base
    prevTangents = tangents

    for i in range(max_iter):
        w_list = []

        for d in range(nManifoldDim):
            for k in range(3):
                w_list.append([])

        for j in range(nData):
            tVec_j = p_anchor.LogMap(Y[j])

            for k in range(3):
                for d in range(nManifoldDim):
                    w_list[k * nManifoldDim + d].append(tVec_j.tVector[k, d])

        estModel_list = []

        for k in range(3):
            for d in range(nManifoldDim):
                t_list_sm = sm.add_constant(new_X)
                w_d_np = np.asarray(w_list[k * nManifoldDim + d])
                LS_model_d = sm.OLS(w_d_np, t_list_sm)
                est_d = LS_model_d.fit(method='qr')

                # est_d = LS_model_d.fit()
                estModel_list.append(est_d)

                # if verbose:
                # 	print( est_d.summary() )

        v_to_base_on_p_anchor = manifolds.kendall3D_tVec(nManifoldDim)
        v_tangent_on_p_anchors = []

        for p in range(new_nParam):
            v_tangent_on_p_anchors.append(manifolds.kendall3D_tVec(nManifoldDim))

        for d in range(nManifoldDim):
            for k in range(3):
                params = estModel_list[nManifoldDim * k + d].params
                v_to_base_on_p_anchor.tVector[k, d] = params[0]
                for p in range(new_nParam):
                    v_tangent_on_p_anchors[p].tVector[k, d] = params[p+1]

        print("Anchor point to base")
        # print( v_to_base_on_p_anchor.tVector )
        print("Step norm: " + str(v_to_base_on_p_anchor.norm()))

        newBase = p_anchor.ExponentialMap(v_to_base_on_p_anchor)
        newTangents = []
        for p in range(new_nParam):
            newTangents.append(p_anchor.ParallelTranslateAtoB(p_anchor, newBase, v_tangent_on_p_anchors[p]))

        energy = 0

        for n in range(nData):
            X_pt = new_X[n]
            target = Y[n]

            current_tangent = manifolds.kendall3D_tVec(nManifoldDim)

            for k in range(3):
                for d in range(nManifoldDim):
                    for p in range(new_nParam):
                        current_tangent.tVector[k, d] += newTangents[p].tVector[k, d] * X_pt[p]

            estimate_n = newBase.ExponentialMap(current_tangent)
            et = estimate_n.LogMap(target)

            # Energy of the tangential error
            energy += et.normSquared()

        if energy >= prevEnergy:
            if verbose:
                print("=========================")
                print("   Energy Increased ")
                print(energy)
                print("=========================")

            break
        else:
            prevBase = newBase
            prevTangents = newTangents
            p_anchor = newBase
            base = newBase
            tangents = newTangents
            prevEnergy = energy
            if verbose:
                print("==================================")
                print(str(i) + "th Iteration energy")
                print(energy)
                print("==================================")
        if stepSize < step_tol:
            if verbose:
                print("==================================")
                print("Step size under tolerance")
                print("Aborting")
                print("==================================")
            break

    return base, tangents, new_X



def MultivariateLinearizedGeodesicPolynomialRegression_Slope_Kendall3D(X, Y, beta0, p0_list, tVec_intercept_arr,
                                                                       cov_intercept_list, level2_order=1, verbose=True):
    # X: list of covariates for slope regression (same as cov_intercept_list for linear cases)
    # Y: list of slopes from level 1 regression
    # tVec_intercept_arr: level 2 tangent vectors from the intercept regression results, which could contains more items
    #                     than the original input covariates due to polynomial expansion
    # level2_order: order of level 2 polynomial regression, doesn't have to be the same as the regression on intercepts

    if verbose:
        print("==============================================================")
        print("  Multivariate Polynomial Regression on Cross Subjects Slopes ")
        print("==============================================================")

    if len(X) == 0 or len(X[0]) == 0:
        print("Covariates cannot be zero-length!")
        return

    nData = len(Y)
    level1_order = len(Y[0])
    nParam = len(X[0])

    # elevate covariates to higher order if needed
    new_X = []
    new_nParam = nParam
    if level2_order == 1:
        for i in range(nData):
            new_X.append(X[i].copy())
    elif level2_order == 2:
        # quadratic expansion
        new_nParam = int((nParam ** 2 + 3 * nParam) / 2)
        for i in range(nData):
            new_x = X[i].copy()
            for j in range(nParam):
                new_x.append(X[i][j] ** 2)
            for j in range(nParam - 1):
                for k in range(j+1, nParam):
                    new_x.append(X[i][j] * X[i][k])
            new_X.append(new_x)
    else:
        raise Exception("The polynomial order of multivariate regression has not been implemented.")

    p_anchor = beta0
    nManifoldDim = p_anchor.nPt

    w_list = []
    for o in range(level1_order):
        w_list.append([])
        for k in range(3):
            for d in range(nManifoldDim):
                w_list[o].append([])

    for j in range(nData):
        # Parallel translate a group-wise tangent vector to population-level intercept
        beta_tVec_f_i = manifolds.kendall3D_tVec(nManifoldDim)

        for tt in range(len(cov_intercept_list[j])):
            est_beta_tt = tVec_intercept_arr[tt]

            for kk in range(3):
                for dd in range(nManifoldDim):
                    beta_tVec_f_i.tVector[kk, dd] += (est_beta_tt.tVector[kk, dd] * cov_intercept_list[j][tt])

        tVec_j = []
        for Y_j in Y[j]:
            f_j = beta0.ExponentialMap(beta_tVec_f_i)
            Y_j_at_f_j = p0_list[j].ParallelTranslateAtoB(p0_list[j], f_j, Y_j)
            Y_j_tilde = f_j.ParallelTranslateAtoB(f_j, beta0, Y_j_at_f_j)

            # if level2_order == 1:
            #     Y_j_tilde = f_j.ParallelTranslateAtoB(f_j, beta0, Y_j_at_f_j)
            # elif level2_order == 2:
            #     # compute normalized base covariates for determining geodesic path on polynomial curve
            #     cov_intercepts = np.zeros(nParam)
            #     for tt in range(nParam):
            #         cov_intercepts[tt] = cov_intercept_list[j][tt]
            #     cov_intercepts /= np.linalg.norm(cov_intercepts)
            #
            #     # tangent vector at f_j
            #     v_f_j = manifolds.kendall3D_tVec(nManifoldDim)
            #     v_f_j_mat = np.zeros([3, nManifoldDim])
            #     # linear terms
            #     for tt in range(nParam):
            #         v_f_j_mat += tVec_intercept_arr[tt].tVector * cov_intercepts[tt]
            #     # quadratic single terms
            #     for tt in range(nParam, 2 * nParam):
            #         v_f_j_mat += 2 * tVec_intercept_arr[tt].tVector * cov_intercept_list[j][tt] * cov_intercepts[tt - nParam]
            #     # quadratic cross terms
            #     offset = 2 * nParam
            #     for tt in range(nParam - 1):
            #         for uu in range(tt + 1, nParam):
            #             v_f_j_mat += tVec_intercept_arr[offset].tVector * cov_intercept_list[j][tt] * cov_intercepts[uu]
            #             v_f_j_mat += tVec_intercept_arr[offset].tVector * cov_intercept_list[j][uu] * cov_intercepts[tt]
            #             offset += 1
            #     v_f_j_mat /= np.linalg.norm(v_f_j_mat)
            #     v_f_j.SetTangentVector(np.asmatrix(v_f_j_mat))
            #
            #     # tangent vector at beta0
            #     v_beta0 = manifolds.kendall3D_tVec(nManifoldDim)
            #     v_beta0_tVec_mat = np.zeros([3, nManifoldDim])
            #     for tt in range(nParam):  # original nParam
            #         v_beta0_tVec_mat += tVec_intercept_arr[tt].tVector * cov_intercepts[tt]
            #     v_beta0_tVec_mat /= np.linalg.norm(v_beta0_tVec_mat)
            #     v_beta0.SetTangentVector(np.asmatrix(v_beta0_tVec_mat))
            #
            #     Y_j_tilde = f_j.ParallelTranslateAtoBCurve(f_j, beta0, Y_j_at_f_j, v_f_j, v_beta0)

            tVec_j.append(Y_j_tilde)

        for o in range(level1_order):
            for k in range(3):
                for d in range(nManifoldDim):
                    w_list[o][k * nManifoldDim + d].append(tVec_j[o].tVector[k, d])

    estModel_list = []
    for o in range(level1_order):
        estModel_list.append([])
        for k in range(3):
            for d in range(nManifoldDim):
                X_sm = sm.add_constant(new_X)
                w_d_np = np.asarray(w_list[o][k * nManifoldDim + d])
                LS_model_d = sm.OLS(w_d_np, X_sm)

                est_d = LS_model_d.fit()
                estModel_list[o].append(est_d)

    # if verbose:
    # 	print( est_d.summary() )

    # base slope for t
    v_t_list = []
    for o in range(level1_order):
        v_t = manifolds.kendall3D_tVec(nManifoldDim)
        for k in range(3):
            for d in range(nManifoldDim):
                v_t.tVector[k, d] = estModel_list[o][k * nManifoldDim + d].params[0]
        v_t_list.append(v_t)

    new_tVec_arr = []
    for o in range(level1_order):
        new_tVec_arr.append([])
        for par in range(new_nParam):
            v_tangent_on_p_anchor_param = manifolds.kendall3D_tVec(nManifoldDim)

            for k in range(3):
                for d in range(nManifoldDim):
                    v_tangent_on_p_anchor_param.tVector[k, d] = estModel_list[o][k * nManifoldDim + d].params[par + 1]

            new_tVec_arr[o].append(v_tangent_on_p_anchor_param)

    # Append time-wise slope tangent vector at the last
    for o in range(level1_order):
        new_tVec_arr[o].append(v_t_list[o])
    tangent_arr = new_tVec_arr

    return tangent_arr, new_X
