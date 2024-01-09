import numbers
import numpy as np
import pickle


class euclidean_tVec(object):
    # def __init__( self ):
    # 	self.Type = "Euclidean_Tangent"
    # 	self.nDim = 3
    # 	self.tVector = [ 0, 0, 0 ]

    def __init__(self, nDim):
        self.Type = "Euclidean_Tangent"
        self.nDim = nDim
        self.tVector = np.zeros(nDim)

    def GetTangentVector(self):
        return self.tVector

    def SetTangentVector(self, tVec):
        if not len(tVec) == self.nDim:
            print("Error : Dimensions does not match")
            return

        self.tVector = tVec

    def InnerProduct(self, tVec1):
        result = 0
        for i in range(self.nDim):
            result += self.tVector[i] * tVec1.tVector[i]
        return result

    def normSquared(self):
        return self.InnerProduct(self)

    def norm(self):
        return np.sqrt(self.normSquared())

    def ScalarMultiply(self, t):
        tVector_t = euclidean_tVec(self.nDim)

        for i in range(self.nDim):
            tVector_t.tVector[i] = self.tVector[i] * t

        return tVector_t

    def Write(self, filePath):
        infoList = [self.Type, self.nDim, self.tVector, False]

        with open(filePath, 'wb') as fp:
            pickle.dump(infoList, fp)

    def Read(self, filePath):
        with open(filePath, 'rb') as fp:
            infoList = pickle.load(fp)

        self.Type = infoList[0]
        self.nDim = infoList[1]
        self.tVector = infoList[2]


class euclidean(object):
    # def __init__( self ):
    # 	self.Type = "Euclidean"
    # 	self.nDim = 3
    # 	self.pt = [ 0.0, 0.0, 0.0 ]

    def __init__(self, nDim):
        self.Type = "Euclidean"
        self.nDim = nDim
        self.pt = np.zeros(nDim)

    def SetPoint(self, pt):
        if not len(pt) == self.nDim:
            print("Error : Dimensions does not match")
            return
        self.pt = pt

    def GetPoint(self):
        return self.pt

    def InnerProduct(self, ptA):
        result = 0
        for i in range(self.nDim):
            result += self.pt[i] * ptA.pt[i]
        return result

    def normSquared(self):
        return self.InnerProduct(self)

    def norm(self):
        return np.sqrt(self.normSquared())

    def ExponentialMap(self, tVec):
        exp_pt = euclidean(self.nDim)

        newPt_mat = np.zeros(self.nDim)

        for i in range(self.nDim):
            newPt_mat[i] = self.pt[i] + tVec.tVector[i]

        exp_pt.SetPoint(newPt_mat.tolist())
        return exp_pt

    def LogMap(self, ptA):
        tVec = euclidean_tVec(self.nDim)

        tVec_list = []

        for i in range(self.nDim):
            tVec_i = ptA.pt[i] - self.pt[i]

            tVec_list.append(tVec_i)

        tVec.SetTangentVector(tVec_list)
        return tVec

    def RiemannianDistanceToA(self, ptA):
        tVec_toA = self.LogMap(ptA)
        distSq = tVec_toA.norm()
        dist = np.sqrt(distSq)
        return dist

    def ProjectTangent(self, pt, tVec):
        vProjected = tVec
        return vProjected

    def ParallelTranslate(self, v, w):
        return w

    def ParallelTranslateToA(self, ptA, w):
        return w

    def ParallelTranslateAtoB(self, ptA, ptB, w):
        return w

    def AdjointGradientJacobi(self, v, j, dj):
        # Function Output
        jOutput = j
        jOutputDash = dj

        jOutputDash_list = []

        for i in range(self.nDim):
            jOutputDash_i = j.tVector[i] + dj.tVector[i]
            jOutputDash_list.append(jOutputDash_i)

        jOutputDash.SetTangentVector(jOutputDash_list)
        return jOutput, jOutputDash

    def Write(self, filePath):
        infoList = [self.Type, self.nDim, self.pt, False]

        with open(filePath, 'wb') as fp:
            pickle.dump(infoList, fp)

    def Read(self, filePath):
        with open(filePath, 'rb') as fp:
            infoList = pickle.load(fp)

        self.Type = infoList[0]
        self.nDim = infoList[1]
        self.pt = infoList[2]


class sphere_tVec(object):
    # def __init__( self ):
    # 	self.Type = "Sphere_Tangent"
    # 	self.nDim = 3
    # 	self.tVector = [ 0, 0, 0 ]

    def __init__(self, nDim):
        self.Type = "Sphere_Tangent"
        self.nDim = nDim
        self.tVector = np.zeros(nDim)

    def GetTangentVector(self):
        return self.tVector

    def SetTangentVector(self, tVec):
        if not len(tVec) == self.nDim:
            print("Error : Dimensions does not match")
            return

        self.tVector = tVec

    def InnerProduct(self, tVec1):
        result = 0

        # print( self.tVector )
        # print( tVec1.tVector )
        # print( self.tVector[ 0 ] )
        # print( tVec1.tVector[ 0 ] )

        for i in range(self.nDim):
            result += self.tVector[i] * tVec1.tVector[i]
        return result

    def normSquared(self):
        return self.InnerProduct(self)

    def norm(self):
        return np.sqrt(self.normSquared())

    def ScalarMultiply(self, t):
        tVector_t = sphere_tVec(self.nDim)

        for i in range(self.nDim):
            tVector_t.tVector[i] = self.tVector[i] * t

        return tVector_t

    def Write(self, filePath):
        infoList = [self.Type, self.nDim, self.tVector, False]

        with open(filePath, 'wb') as fp:
            pickle.dump(infoList, fp)

    def Read(self, filePath):
        with open(filePath, 'rb') as fp:
            infoList = pickle.load(fp)

        self.Type = infoList[0]
        self.nDim = infoList[1]
        self.tVector = infoList[2]


class sphere(object):
    # def __init__( self ):
    # 	self.Type = "Sphere"
    # 	self.nDim = 3
    # 	self.pt = [ 1.0, 0.0, 0.0 ] # Base point in S^2

    def __init__(self, nDim):
        self.Type = "Sphere"
        self.nDim = nDim
        pt_base = np.zeros(nDim)
        pt_base[0] = 1
        self.pt = pt_base

    def SetPoint(self, pt):
        if not len(pt) == self.nDim:
            print("Error : Dimensions does not match")
            return

        if not np.linalg.norm(pt) == 1:
            # print( "Warning : The point is not on a sphere")
            self.pt = pt
            return

        self.pt = pt

    def GetPoint(self):
        return self.pt

    def InnerProduct(self, ptA):
        result = 0
        for i in range(self.nDim):
            result += self.pt[i] * ptA.pt[i]
        return result

    def normSquared(self):
        return self.InnerProduct(self)

    def norm(self):
        return np.sqrt(self.normSquared())

    def ProjectTangent(self, pt, tVec):
        inner_prod_pt_tVec = 0
        for i in range(self.nDim):
            inner_prod_pt_tVec += pt.pt[i] * tVec.tVector[i]

        vProjected = sphere_tVec(self.nDim)

        for i in range(self.nDim):
            vProjected.tVector[i] = tVec.tVector[i] - inner_prod_pt_tVec * pt.pt[i]

        return vProjected

    def ParallelTranslate(self, v, w):
        vNorm = v.norm()

        if (vNorm < 1.0e-12):
            return w

        innerProd = v.InnerProduct(w)
        scaleFactor = innerProd / (vNorm * vNorm)

        # Component of w orthogonal to v
        orth = sphere_tVec(self.nDim)
        for i in range(self.nDim):
            orth.tVector[i] = w.tVector[i] - v.tVector[i] * scaleFactor

        # Compute parallel translated v
        vParallel = sphere_tVec(self.nDim)

        for i in range(self.nDim):
            vParallel.tVector[i] = self.pt[i] * (-np.sin(vNorm) * vNorm) + v.tVector[i] * np.cos(vNorm)

        # Parallel Translated w
        wParallelTranslated = sphere_tVec(self.nDim)

        for i in range(self.nDim):
            wParallelTranslated.tVector[i] = vParallel.tVector[i] * scaleFactor + orth.tVector[i]

        return wParallelTranslated

    def ParallelTranslateToA(self, ptA, w):
        v = self.LogMap(ptA)
        vNorm = v.norm()

        if (vNorm < 1.0e-12):
            return w

        innerProd = v.InnerProduct(w)
        scaleFactor = innerProd / (vNorm * vNorm)

        # Component of w orthogonal to v
        orth = sphere_tVec(self.nDim)
        for i in range(self.nDim):
            orth.tVector[i] = w.tVector[i] - v.tVector[i] * scaleFactor

        # Compute parallel translated v
        vParallel = sphere_tVec(self.nDim)

        for i in range(self.nDim):
            vParallel.tVector[i] = self.pt[i] * (-np.sin(vNorm) * vNorm) + v.tVector[i] * np.cos(vNorm)

        # Parallel Translated w
        wParallelTranslated = sphere_tVec(self.nDim)

        for i in range(self.nDim):
            wParallelTranslated.tVector[i] = vParallel.tVector[i] * scaleFactor + orth.tVector[i]

        return wParallelTranslated

    def AdjointGradientJacobi(self, v, j, dj):
        e_base = self

        vNorm = v.norm()

        if vNorm < 1.0e-12:
            for i in range(self.nDim):
                dj.tVector[i] = j.tVector[i] + dj.tVector[i]

            # Function Output
            jOutput = j
            jOutputDash = dj
        else:
            innerProdVJ = v.InnerProduct(j)
            innerProdVJPrime = v.InnerProduct(dj)

            scaleFactorJ = innerProdVJ / (vNorm * vNorm)
            scaleFactorJPrime = innerProdVJPrime / (vNorm * vNorm)

            jTang = sphere_tVec(self.nDim)
            djTang = sphere_tVec(self.nDim)
            jOrth = sphere_tVec(self.nDim)
            djOrth = sphere_tVec(self.nDim)

            for i in range(self.nDim):
                jTang.tVector[i] = v.tVector[i] * scaleFactorJ
                djTang.tVector[i] = v.tVector[i] * scaleFactorJPrime

                jOrth.tVector[i] = j.tVector[i] - jTang.tVector[i]
                djOrth.tVector[i] = dj.tVector[i] - djTang.tVector[i]

                j.tVector[i] = jTang.tVector[i] + (np.cos(vNorm) * jOrth.tVector[i]) - (
                            (vNorm * np.sin(vNorm)) * djOrth.tVector[i])

            j = e_base.ParallelTranslate(v, j)

            for i in range(self.nDim):
                dj.tVector[i] = jTang.tVector[i] + djTang.tVector[i] + (np.sin(vNorm) / vNorm) * jOrth.tVector[
                    i] + np.cos(vNorm) * djOrth.tVector[i]

            dj = e_base.ParallelTranslate(v, dj)

            # Function Output
            jOutput = j
            jOutputDash = dj

        return jOutput, jOutputDash

    def ParallelTranslateAtoB(self, a, b, w):
        v = a.LogMap(b)
        return a.ParallelTranslate(v, w)

    def ExponentialMap(self, tVec):
        theta = np.linalg.norm(tVec.tVector)

        if theta < 1e-12:
            exp_pt = sphere(self.nDim)
            exp_pt.pt = self.pt
            return exp_pt

        if theta > np.pi * 2:
            theta = np.mod(theta, np.pi * 2)

        exp_pt = sphere(self.nDim)

        lhs = np.multiply(np.cos(theta), self.pt)
        rhs = np.multiply(np.sin(theta) / theta, tVec.tVector)

        exp_pt.pt = lhs + rhs
        exp_pt.pt = np.divide(exp_pt.pt, exp_pt.norm())

        return exp_pt

    def LogMap(self, another_pt):
        cosTheta = self.InnerProduct(another_pt)
        tVec = sphere_tVec(self.nDim)

        for i in range(self.nDim):
            tVec.tVector[i] = another_pt.pt[i] - cosTheta * self.pt[i]

        length = tVec.norm()
        if length < 1e-12 or cosTheta >= 1.0 or cosTheta <= -1.0:
            tVec = sphere_tVec(self.nDim)
            return tVec

        for i in range(self.nDim):
            tVec.tVector[i] = tVec.tVector[i] * np.arccos(cosTheta) / length

        return tVec

    def Write(self, filePath):
        infoList = [self.Type, self.nDim, self.pt, False]

        with open(filePath, 'wb') as fp:
            pickle.dump(infoList, fp)

    def Read(self, filePath):
        with open(filePath, 'rb') as fp:
            infoList = pickle.load(fp)

        self.Type = infoList[0]
        self.nDim = infoList[1]
        self.pt = infoList[2]


class pos_real_tVec(object):
    def __init__(self, nDim):
        self.Type = "PositiveReal_Tangent"
        self.nDim = nDim
        self.tVector = np.zeros(nDim).tolist()

    def GetTangentVector(self):
        return self.tVector

    def SetTangentVector(self, tVec):
        if type(tVec) == list:
            if not len(tVec) == self.nDim:
                print("Error : Dimensions does not match")
                return
            else:
                self.tVector = tVec
        else:
            if not self.nDim == 1:
                print("Error : Dimensions does not match")
                return
            else:
                self.tVector[0] = tVec

    def InnerProduct(self, tVec1):
        result = 0
        for i in range(self.nDim):
            result += self.tVector[i] * tVec1.tVector[i]
        return result

    def normSquared(self):
        return self.InnerProduct(self)

    def norm(self):
        return np.sqrt(self.normSquared())

    def ScalarMultiply(self, t):
        tVector_t = pos_real_tVec(self.nDim)

        for i in range(self.nDim):
            tVector_t.tVector[i] = self.tVector[i] * t

        return tVector_t

    def Write(self, filePath):
        infoList = [self.Type, self.nDim, self.tVector, False]

        with open(filePath, 'wb') as fp:
            pickle.dump(infoList, fp)

    def Read(self, filePath):
        with open(filePath, 'rb') as fp:
            infoList = pickle.load(fp)

        self.Type = infoList[0]
        self.nDim = infoList[1]
        self.tVector = infoList[2]


class pos_real(object):
    def __init__(self, nDim):
        self.Type = "PositiveReal"
        self.nDim = nDim
        self.pt = np.ones(nDim).tolist()

    def SetPoint(self, pt):
        if type(pt) == list:
            if not len(pt) == self.nDim:
                print("Error : Dimensions does not match")
                return
            else:
                self.pt = pt
        else:
            if not self.nDim == 1:
                print("Error : Dimensions does not match")
                return
            else:
                self.pt[0] = pt

    def GetPoint(self):
        return self.pt

    def InnerProduct(self, ptA):
        result = 0
        for i in range(self.nDim):
            result += self.pt[i] * ptA.pt[i]
        return result

    def normSquared(self):
        return self.InnerProduct(self)

    def norm(self):
        return np.sqrt(self.normSquared())

    def ExponentialMap(self, tVec):
        exp_pt = pos_real(self.nDim)
        # print( "Tangent Vector" )
        # print( tVec.tVector )
        exp_pt.SetPoint(np.multiply(self.pt, np.exp(tVec.tVector)).tolist())
        return exp_pt

    def LogMap(self, another_pt):
        tVec = pos_real_tVec(self.nDim)

        tVec.SetTangentVector(np.log(np.divide(another_pt.pt, self.pt)).tolist())
        return tVec

    def ProjectTangent(self, pt, tVec):
        vProjected = tVec
        return vProjected

    def ParallelTranslate(self, v, w):
        return w

    def ParallelTranslateToA(self, ptA, w):
        return w

    def ParallelTranslateAtoB(self, a, b, w):
        return w

    def AdjointGradientJacobi(self, v, j, dj, ):
        jOutput = j
        jOutputDash = dj

        for i in range(self.nDim):
            jOutputDash.tVector[i] = j.tVector[i] + dj.tVector[i]

        return jOutput, jOutputDash

    def Write(self, filePath):
        infoList = [self.Type, self.nDim, self.pt, False]

        with open(filePath, 'wb') as fp:
            pickle.dump(infoList, fp)

    def Read(self, filePath):
        with open(filePath, 'rb') as fp:
            infoList = pickle.load(fp)

        self.Type = infoList[0]
        self.nDim = infoList[1]
        self.pt = infoList[2]

    ###############################################################


#####					 Miscelleneous		 			  #####
###############################################################
def HelmertSubmatrix(nAtoms):
    # Create a Helmert submatrix - similarity-invariant
    H = np.zeros([nAtoms - 1, nAtoms])

    for k in range(nAtoms - 1):
        h_k = -np.divide(1.0, np.sqrt((k + 1) * (k + 2)))
        neg_kh_k = np.multiply(h_k, -(k + 1))
        for h in range(k + 1):
            H[k, h] = h_k
        H[k, k + 1] = neg_kh_k

    return H


def HelmertMatrix(nAtoms):
    # Create a Helmert matrix - similiarity-invariant : First row - Center of Gravity (mass) (uniform mass of points)
    H_full = np.zeors([nAtoms, nAtoms])

    for h in range(nAtoms):
        H_full[0, h] = np.divide(1, np.sqrt(nAtoms))

    for k in range(1, nAtoms, 1):
        h_k = -np.divide(1.0, np.sqrt((k) * (k + 1)))
        neg_kh_k = np.multiply(h_k, -k)
        for h in range(k):
            H_full[k, h] = h_k
        H_full[k, k] = neg_kh_k

    return H_full


##########################################################################
## 							Kendall 2D Shape Space					    ##
##########################################################################

class kendall2D_tVec(object):
    # def __init__( self ):
    # 	self.Type = "Sphere_Tangent"
    # 	self.nDim = 3
    # 	self.tVector = [ 0, 0, 0 ]

    def __init__(self, nPt):
        self.Type = "Kendall2D_Tangent"
        self.nPt = nPt
        self.nDim = nPt - 2
        self.tVector = np.zeros([2, nPt])

    def GetTangentVector(self):
        return self.tVector

    def SetTangentVector(self, tVec):
        if not tVec.shape[1] == self.nPt:
            print("Error : # of points does not match")
            return

        if not tVec.shape[0] == 2:
            print("Error : Tangent vector should be 2D")
            return

        self.tVector = tVec

    def InnerProduct(self, tVec1):
        result = 0
        for i in range(self.nPt):
            for j in range(2):
                result += self.tVector[j, i] * tVec1.tVector[j, i]
        return result

    def normSquared(self):
        return self.InnerProduct(self)

    def norm(self):
        return np.sqrt(self.normSquared())

    def ScalarMultiply(self, t):
        tVector_t = kendall2D_tVec(self.nPt)

        for i in range(self.nPt):
            for j in range(2):
                tVector_t.tVector[j, i] = self.tVector[j, i] * t

        return tVector_t

    def Write(self, filePath):
        infoList = [self.Type, self.nDim, self.nPt, self.tVector, False]

        with open(filePath, 'wb') as fp:
            pickle.dump(infoList, fp)

    def Read(self, filePath):
        with open(filePath, 'rb') as fp:
            infoList = pickle.load(fp)

        self.Type = infoList[0]
        self.nDim = infoList[1]
        self.nPt = infoList[2]
        self.tVector = infoList[3]


class kendall2D(object):
    def __init__(self, nPt):
        self.Type = "Kendall2D"
        self.nPt = nPt
        self.nDim = nPt - 2

        pt_base = np.zeros([2, nPt])
        pt_base[0, 0] = 1
        pt_base[0, 1] = 0

        self.pt = pt_base

    def SetPoint(self, pt):
        if not pt.shape[1] == self.nPt:
            print("Error : # of Points does not match")
            return
        if not pt.shape[0] == 2:
            print("Error : Point should be 2D")
            return

        if not np.linalg.norm(pt) == 1:
            # print( "Warning : The point is not on a sphere")
            self.pt = np.asmatrix(pt)
            return

        self.pt = np.asmatrix(pt)

    def GetPoint(self):
        return self.pt

    def InnerProduct(self, ptA):
        result = 0
        for i in range(self.nPt):
            for j in range(2):
                result += self.pt[j, i] * ptA.pt[j, i]
        return result

    def normSquared(self):
        return self.InnerProduct(self)

    def norm(self):
        return np.sqrt(self.normSquared())

    def ExponentialMap(self, tVec):
        theta = tVec.norm()

        if theta < 1e-12:
            exp_pt = kendall2D(self.nPt)
            exp_pt.pt = self.pt
            return exp_pt

        if theta > np.pi * 2:
            theta = np.mod(theta, np.pi * 2)

        exp_pt = kendall2D(self.nPt)

        lhs = np.multiply(np.cos(theta), self.pt)
        rhs = np.multiply(np.sin(theta) / theta, tVec.tVector)

        exp_pt.pt = lhs + rhs
        exp_pt.pt = np.divide(exp_pt.pt, exp_pt.norm())

        return exp_pt

    def LogMap(self, another_pt):
        m = np.matmul(self.pt, another_pt.pt.T)

        U, s, V = np.linalg.svd(m)

        rotation = np.matmul(U, V.T)

        qRot_pt = np.matmul(rotation, another_pt.pt)

        qRot = kendall2D(self.nPt)
        qRot.SetPoint(qRot_pt)

        cosTheta = self.InnerProduct(qRot)

        tVec = kendall2D_tVec(self.nPt)

        tVec_mat = np.subtract(qRot.pt, np.multiply(cosTheta, self.pt))
        tVec.SetTangentVector(tVec_mat)

        length = tVec.norm()

        if length < 1e-12 or cosTheta >= 1.0 or cosTheta <= -1.0:
            tVec = kendall2D_tVec(self.nPt)
            return tVec

        tVec = tVec.ScalarMultiply(np.arccos(cosTheta) / length)

        return tVec

    def ParallelTranslate(self, v, w):
        vNorm = v.norm()
        pNorm = self.norm()

        if (vNorm < 1.0e-12 or pNorm < 1.0e-12):
            # print( "tVector too small" )
            return w

        skew = np.zeros([2, 2])
        skew[0, 1] = -1
        skew[1, 0] = 1

        unitV = v.ScalarMultiply(1.0 / vNorm)

        unitJV_mat = np.matmul(skew, unitV.tVector)

        unitJV = kendall2D_tVec(self.nPt)
        unitJV.SetTangentVector(unitJV_mat)

        unitP = self.ScalarMultiply(1.0 / pNorm)

        unitJP_mat = np.matmul(skew, unitP.pt)
        unitJP = kendall2D(self.nPt)
        unitJP.SetPoint(unitJP_mat)

        # If v and w are horizontal, the real inner product will work
        wDotUnitV = unitV.InnerProduct(w)
        wDotUnitJV = unitJV.InnerProduct(w)

        # Component of w orthogonal to v and jv
        parallel_mat = np.add(np.multiply(wDotUnitV, unitV.tVector), np.multiply(wDotUnitJV, unitJV_mat))

        orth_mat = np.subtract(w.tVector, parallel_mat)

        # Compute Parallel Translated V
        parallelUnitV_mat = np.add(np.multiply(self.pt, -np.sin(vNorm) / pNorm),
                                   np.multiply(np.cos(vNorm), unitV.tVector))

        # Compute Parallel Translated jV
        parallelUnitJV_mat = np.subtract(np.multiply(np.cos(vNorm), unitJV_mat), np.multiply(np.sin(vNorm), unitJP_mat))

        # Add parallel translated v to orth, and get parallel translated w
        parallelW_paraV = np.add(np.multiply(wDotUnitV, parallelUnitV_mat), np.multiply(wDotUnitJV, parallelUnitJV_mat))

        parallelW_mat = np.add(parallelW_paraV, orth_mat)

        wParallelTranslated = kendall2D_tVec(self.nPt)
        wParallelTranslated.SetTangentVector(parallelW_mat)

        return wParallelTranslated

    def ParallelTranslateAtoB(self, a, b, w):
        v = a.LogMap(b)
        return a.ParallelTranslate(v, w)

    def ParallelTranslateToA(self, a, w):
        v = self.LogMap(a)
        return self.ParallelTranslate(v, w)

    def ScalarMultiply(self, t):
        p_t = kendall2D(self.nPt)

        for i in range(self.nPt):
            for j in range(2):
                p_t.pt[j, i] = self.pt[j, i] * t

        return p_t

    def GradientJacobi(self, v, J, dJ):
        vNorm = v.norm()

        if (vNorm < 1.0e-12):
            for i in range(self.nPt):
                for k in range(2):
                    J.tVector[k][i] = J.tVector[k][i] + dJ.tVector[k][i]

            return J

        VdotJ = v.InnerProduct(J)
        VdotJPrime = v.InnerProduct(dJ)

        scaleFactorJ = VdotJ / (vNorm * vNorm)
        scaleFactorJPrime = VdotJPrime / (vNorm * vNorm)

        jTang_mat = np.multiply(v.tVector, scaleFactorJ)
        jTang = kendall2D_tVec(self.nPt)
        jTang.SetTangentVector(jTang_mat)

        dJTang_mat = np.multiply(v.tVector, scaleFactorJPrime)
        dJTang = kendall2D_tVec(self.nPt)
        dJTang.SetTangentVector(dJTang_mat)

        jOrth_mat = np.subtract(J.tVector, jTang_mat)
        jOrth = kendall2D_tVec(self.nPt)
        jOrth.SetTangentVector(jOrth_mat)

        dJOrth_mat = np.subtract(dJ.tVector, dJTang_mat)
        dJOrth = kendall2D_tVec(self.nPt)
        dJOrth.SetTangentVector(dJOrth_mat)

        skew = np.zeros([2, 2])
        skew[0, 1] = -1
        skew[1, 0] = 1

        unitV = v.ScalarMultiply(1.0 / vNorm)

        w_mat = np.matmul(skew, unitV.tVector)
        w = kendall2D_tVec(self.nPt)
        w.SetTangentVector(w_mat)

        # Curvature 4 component
        jOrth4 = w.ScalarMultiply(w.InnerProduct(jOrth))
        dJOrth4 = w.ScalarMultiply(w.InnerProduct(dJOrth))

        # Curvature 1 Component
        jOrth1 = kendall2D_tVec(self.nPt)
        jOrth1.SetTangentVector(np.subtract(jOrth.tVector, jOrth4.tVector))

        dJOrth1 = kendall2D_tVec(self.nPt)
        dJOrth1.SetTangentVector(np.subtract(dJOrth.tVector, dJOrth4.tVector))

        # Orthogonal Parts
        jOrth.SetTangentVector(
            np.add(np.multiply(cos(vNorm), jOrth1.tVector), np.multiply(cos(2.0 * vNorm), jOrth4.tVector)))

        dJOrth.SetTangentVector(np.add(np.multiply(np.sin(vNorm) / vNorm, dJOrth1.tVector),
                                       np.multiply(0.5 * np.sin(2.0 * vNorm) / vNorm, dJOrth4.tVector)))

        J_dJ_mat = jTang.tVector + dJTang.tVector + jOrth.tVector + dJOrth.tVector
        J_dJ = kendall2D_tVec(self.nPt)
        J_dJ.SetTangentVector(J_DJ)

        J = self.ParallelTranslate(v, J_dJ)

        dJOrth_mat = jOrth1.ScalarMultiply(-vNorm * np.sin(vNorm)).tVector + jOrth4.ScalarMultiply(
            -2.0 * vNorm * sin(2.0 * vNorm)).tVector

        dJOrth.SetTangentVector(dJOrth_mat)

        ddJOrth_mat = dJOrth1.ScalarMultiply(cos(vNorm)).tVector + djOrth4.ScalarMultiply(cos(2.0 * vNorm)).tVector

        ddJOrth = kendall2D_tVec(self.nPt)
        ddJOrth.SetTangentVector(ddJOrth_mat)

        dJ_ddJ_mat = djTang.tVector + dJOrth.tVector + ddJOrth.tVector

        dJ_ddJ = kendall2D_tVec(self.nPt)

        dJ = self.ParallelTranslate(v, dJ_ddJ)

        return J, dJ

    def AdjointGradientJacobi(self, v, Jac, dJac):
        vNorm = v.norm()

        if (vNorm < 1.0e-12):
            for i in range(self.nPt):
                for j in range(2):
                    Jac.tVector[j][i] = Jac.tVector[j][i] + dJac.tVector[j][i]
            Jac_Updated = Jac
            dJac_Updated = dJac

            return Jac_Updated, dJac_Updated

        VdotJac = v.InnerProduct(Jac)
        VdotJacPrime = v.InnerProduct(dJac)

        scaleFactorJac = VdotJac / (vNorm * vNorm)
        scaleFactorJacPrime = VdotJacPrime / (vNorm * vNorm)

        jTang_mat = np.multiply(v.tVector, scaleFactorJac)
        jTang = kendall2D_tVec(self.nPt)
        jTang.SetTangentVector(jTang_mat)

        dJacTang_mat = np.multiply(v.tVector, scaleFactorJacPrime)
        dJacTang = kendall2D_tVec(self.nPt)
        dJacTang.SetTangentVector(dJacTang_mat)

        jOrth_mat = np.subtract(Jac.tVector, jTang_mat)
        jOrth = kendall2D_tVec(self.nPt)
        jOrth.SetTangentVector(jOrth_mat)

        dJacOrth_mat = np.subtract(dJac.tVector, dJacTang_mat)
        dJacOrth = kendall2D_tVec(self.nPt)
        dJacOrth.SetTangentVector(dJacOrth_mat)

        skew = np.zeros([2, 2])
        skew[0, 1] = -1
        skew[1, 0] = 1

        unitV = v.ScalarMultiply(1.0 / vNorm)

        w_mat = np.matmul(skew, unitV.tVector)
        w = kendall2D_tVec(self.nPt)
        w.SetTangentVector(w_mat)

        # Curvature 4 component
        jOrth4 = w.ScalarMultiply(w.InnerProduct(jOrth))
        dJacOrth4 = w.ScalarMultiply(w.InnerProduct(dJacOrth))

        # Curvature 1 Component
        jOrth1 = kendall2D_tVec(self.nPt)
        jOrth1.SetTangentVector(np.subtract(jOrth.tVector, jOrth4.tVector))

        dJacOrth1 = kendall2D_tVec(self.nPt)
        dJacOrth1.SetTangentVector(np.subtract(dJacOrth.tVector, dJacOrth4.tVector))

        # Orthogonal Parts
        jOrth.SetTangentVector(
            np.add(np.multiply(np.cos(vNorm), jOrth1.tVector), np.multiply(np.cos(2.0 * vNorm), jOrth4.tVector)))

        dJacOrth.SetTangentVector(np.add(np.multiply(-vNorm * np.sin(vNorm), dJacOrth1.tVector),
                                         np.multiply(-2.0 * vNorm * np.sin(2.0 * vNorm), dJacOrth4.tVector)))

        Jac_dJac_mat = jTang.tVector + jOrth.tVector + dJacOrth.tVector
        Jac_dJac = kendall2D_tVec(self.nPt)
        Jac_dJac.SetTangentVector(Jac_dJac_mat)

        Jac_Updated = self.ParallelTranslate(v, Jac_dJac)

        dJacOrth_mat = jOrth1.ScalarMultiply(np.sin(vNorm) / vNorm).tVector + jOrth4.ScalarMultiply(
            0.5 * np.sin(2.0 * vNorm) / vNorm).tVector

        dJacOrth.SetTangentVector(dJacOrth_mat)

        ddJacOrth_mat = dJacOrth1.ScalarMultiply(np.cos(vNorm)).tVector + dJacOrth4.ScalarMultiply(
            np.cos(2.0 * vNorm)).tVector

        ddJacOrth = kendall2D_tVec(self.nPt)
        ddJacOrth.SetTangentVector(ddJacOrth_mat)

        dJac_ddJac_mat = jTang.tVector + dJacTang.tVector + dJacOrth.tVector + ddJacOrth.tVector

        dJac_ddJac = kendall2D_tVec(self.nPt)
        dJac_Updated = self.ParallelTranslate(v, dJac_ddJac)

        return Jac_Updated, dJac_Updated

    def CurvatureTensor(self, x, y, z):
        skew = np.zeros([2, 2])
        skew[0, 1] = -1
        skew[1, 0] = 1

        jX_mat = np.matmul(skew, x.tVector)
        jY_mat = np.matmul(skew, y.tVector)
        jZ_mat = np.matmul(skew, z.tVector)

        jX = kendall2D_tVec(self.nPt)
        jX.SetTangentVector(jX_mat)

        jY = kendall2D_tVec(self.nPt)
        jY.SetTangentVector(jY_mat)

        jZ = kendall2D_tVec(self.nPt)
        jZ.SetTangentVector(jZ_mat)

        zxy_mat = np.multiply(self.InnerProduct(z, jX), jY)
        xyz_mat = np.multiply(self.InnerProduct(x, jY), jZ)
        yzx_mat = np.multiply(self.InnerProduct(y, jZ), jX)

        kCurv_mat = zxy_mat + yzx_mat - (np.multiply(2.0, xyz))

        zDotX = self.InnerProduct(z, x)
        zDotY = self.InnerProduct(z, y)

        sphereCurv_mat = np.multiply(zDotX, y.tVector) - np.multiply(zDotY, x.tVector)

        curv_mat = kCurv_mat + sphereCurv_mat

        curv = kendall2D_tVec(self.nPt)
        curv.SetTangentVector(curv_mat)

        return curv

    def SectionalCurvature(self, x, y):
        curv = self.CurvatureTensor(x, y, x)

        sec = self.InnerProduct(curv, y)
        xx = self.normSquared(x)
        yy = self.normSquared(y)
        xy = self.InnerProduct(x, y)

        secCurv = sec / ((xx * yy) - (xy ** 2))
        return secCurv

    def ProjectTangent(self, p, v):
        meanV = np.average(v.tVector, axis=1)

        print(meanV.shape)

        hV = v

        for i in range(self.nPt):
            for j in range(2):
                hv.tVector[j, i] = v.tVector[j, i] - meanV[j]

        skew = np.zeros([2, 2])
        skew[0, 1] = -1
        skew[1, 0] = 1

        vert_mat = np.matmul(skew, p.pt)
        vert = kendall2D_tVec(self.nPt)
        vert.SetTangentVector(vert_mat)

        new_hV_mat = hV.tVector - np.multiply(p.InnerProduct(hV, vert), vert.tVector)

        hV.SetTangentVector(new_hV_mat)

        return hV

    def Write(self, filePath):
        infoList = [self.Type, self.nDim, self.nPt, self.pt, False]

        with open(filePath, 'wb') as fp:
            pickle.dump(infoList, fp)

    def Read(self, filePath):
        with open(filePath, 'rb') as fp:
            infoList = pickle.load(fp)

        self.Type = infoList[0]
        self.nDim = infoList[1]
        self.nPt = infoList[2]
        self.pt = infoList[3]


##########################################################################
##  					Scale Kendall 2D Shape Space					##
##########################################################################
class scale_kendall2D(object):
    def __init__(self, nPt):
        self.Type = "Scale_Kendall2D"
        self.nPt = nPt
        self.nDim = nPt - 2
        # pt : center, scale, abstract position, radius
        self.pt = [euclidean(1), kendall2D(nPt)]
        self.scale = self.pt[0]
        self.kShape = self.pt[1]
        self.meanScale = 0

    def SetPoint(self, pt):
        if not (len(pt) == 2 and pt[0].nDim == 1 and pt[1].nPt == self.nPt):
            print("Scale_Kendall2D.SetPoint")
            print("Error : Dimensions does not match")
            return
        self.pt = pt
        self.scale = pt[0]
        self.kShape = pt[1]

    def SetMeanScale(self, s):
        self.meanScale = s
        return

    def ExponentialMap(self, tVec):
        if not tVec.Type == "Scale_Kendall2D_Tangent":
            print("Tangent Vector Type Mismatched")
            return

        exp_pt = scale_kendall2D(self.nPt)
        exp_pt_arr = []

        for i in range(2):
            exp_pt_arr.append(self.pt[i].ExponentialMap(tVec.tVector[i]))

        exp_pt.SetPoint(exp_pt_arr)

        return exp_pt

    def LogMap(self, another_pt):
        if not another_pt.Type == "Scale_Kendall2D":
            print("Error: Component Type Mismatched")
            return

        tVec = scale_kendall2D_tVec(self.nPt)

        for i in range(2):
            tVec.tVector[i] = self.pt[i].LogMap(another_pt.pt[i])

        return tVec

    def GetScale(self):
        return self.scale

    def GetShape(self):
        return self.kShape

    def InnerProduct(self, ptA):
        result = 0

        # Scale
        result += self.pt[0].InnerProduct(ptA.pt[0])

        # Abstract Position
        result += self.pt[1].InnerProduct(ptA.pt[1])

        return result

    def normSquared(self):
        return self.InnerProduct(self)

    def norm(self):
        return np.sqrt(self.normSquared())

    def ProjectTangent(self, pt, tVec):
        vProjected = scale_kendall2D_tVec(self.nPt)

        for i in range(2):
            vProjected_tVector[i] = self.pt[i].ProjectTangent(pt.pt[i], tVec.tVector[i])

        return vProjected

    def ParallelTranslate(self, v, w):
        wParallelTranslated = scale_kendall2D_tVec(self.nDim)

        for i in range(2):
            wParallelTranslated.tVector[i] = self.pt[i].ParallelTranslate(v.tVector[i], w.tVector[i])

        return wParallelTranslated

    def ParallelTranslateToA(self, ptA, w):
        v = self.LogMap(ptA)
        return ParallelTranslate(v, w)

    def ParallelTranslateAtoB(self, a, b, w):
        v = a.LogMap(b)
        return a.ParallelTranslate(v, w)

    def AdjointGradientJacobi(self, v, j, dj):
        e_base = self
        vNorm = v.norm()
        jOutput = scale_kendall2D_tVec(self.nPt)
        jOutputDash = scale_kendall2D_tVec(self.nPt)

        for i in range(2):
            jOutput.tVector[i], jOutputDash.tVector[i] = self.pt[i].AdjointGradientJacobi(v.tVector[i], j.tVector[i],
                                                                                          dj.tVector[i])

        return jOutput, jOutputDash

    def GetEuclideanLocations(self):
        # Multiply Scale
        pos_scale_eucldiean_matrix = np.multiply(self.kShape.pt, self.scale.pt[0])

        # Add Center of Mass
        pos_world_coord_euclidean_matrix = np.zeros([self.nPt, 2])

        for i in range(self.nPt):
            pos_world_coord_euclidean_matrix[i, :] = pos_scale_eucldiean_matrix[:, i].flatten()

        return pos_world_coord_euclidean_matrix

    def Write(self, filePath):
        infoList = [self.Type, self.nPt, self.nDim, self.pt, self.meanScale]

        with open(filePath, 'wb') as fp:
            pickle.dump(infoList, fp)

    def Read(self, filePath):
        with open(filePath, 'rb') as fp:
            infoList = pickle.load(fp)

        self.Type = infoList[0]
        self.nPt = infoList[1]
        self.nDim = infoList[2]
        self.pt = infoList[3]

        self.scale = self.pt[0]
        self.kShape = self.pt[1]

        self.meanScale = infoList[4]


class scale_kendall2D_tVec(object):
    def __init__(self, nPt):
        self.Type = "Scale_Kendall2D_Tangent"
        self.nPt = nPt
        self.nDim = nPt - 2
        # tVector : scale, 2D Kendall Shape
        self.tVector = [euclidean_tVec(1), kendall2D_tVec(nPt)]
        self.meanScale = 1

    # def SetTangentVectorFromArray( self, tVecArr ):
    # 	if not len( tVecArr ) == ( 4 + self.tVector[ 2 ].nDim + self.tVector[ 3 ].nDim ):
    # 		print( "Error : Dimension Mismatch" )
    # 		return

    # 	# Center
    # 	self.tVector[ 0 ].tVector[ 0 ] = tVecArr[ 0 ]
    # 	self.tVector[ 0 ].tVector[ 1 ] = tVecArr[ 1 ]
    # 	self.tVector[ 0 ].tVector[ 2 ] = tVecArr[ 2 ]

    # 	# Scale
    # 	self.tVector[ 1 ].tVector[ 0 ] = tVecArr[ 3 ]

    # 	# PreShape
    # 	for k in range( self.tVector[ 2 ].nDim ):
    # 		self.tVector[ 2 ].tVector[ k ] = tVecArr[ k + 4 ]

    # 	# Radius
    # 	for k in range( self.tVector[ 3 ].nDim ):
    # 		self.tVector[ 3 ].tVector[ k ] = tVecArr[ k + 4 + self.tVector[ 2 ].nDim ]

    def SetMeanScale(self, meanScale):
        self.meanScale = meanScale
        return

    def GetTangentVector(self):
        return self.tVector

    def SetTangentVector(self, tVec):
        if not (len(tVec) == 2 and tVec[0].nDim == 1 and tVec[1].nPt == self.nPt):
            print("Error : Dimension Mismatch")
            return

        self.tVector = tVec

    def InnerProduct(self, tVec1):
        result = 0

        # Scale
        result += self.tVector[0].InnerProduct(tVec1.tVector[0])

        # Kendall Shapes
        result += self.meanScale * self.meanScale * self.tVector[1].InnerProduct(tVec1.tVector[1])

        return result

    def normSquared(self):
        return self.InnerProduct(self)

    def norm(self):
        return np.sqrt(self.normSquared())

    def ScalarMultiply(self, t):
        tVector_t = scale_kendall2D_tVec(self.nPt)

        for i in range(2):
            tVector_t.tVector[i] = self.tVector[i].ScalarMultiply(t)

        return tVector_t

    def Write(self, filePath):
        infoList = [self.Type, self.nPt, self.nDim, self.tVector, self.meanScale]

        with open(filePath, 'wb') as fp:
            pickle.dump(infoList, fp)

    def Read(self, filePath):
        with open(filePath, 'rb') as fp:
            infoList = pickle.load(fp)

        self.Type = infoList[0]
        self.nPt = infoList[1]
        self.nDim = infoList[2]
        self.tVector = infoList[3]
        self.meanScale = infoList[4]


##########################################################################
## 							Kendall 3D Shape Space					    ##
##########################################################################


class kendall3D_tVec(object):

    def __init__(self, nPt):
        self.Type = "Kendall3D_Tangent"
        self.nPt = nPt
        self.nDim = nPt - 2
        self.tVector = np.zeros([3, nPt])

    def GetTangentVector(self):
        return self.tVector

    def SetTangentVector(self, tVec):
        if not tVec.shape[1] == self.nPt:
            print("Error : # of points does not match")
            return

        if not tVec.shape[0] == 3:
            print("Error : Tangent vector should be 3D")
            return

        self.tVector = tVec

    def InnerProduct(self, tVec1):
        result = 0
        for i in range(self.nPt):
            for j in range(3):
                result += self.tVector[j, i] * tVec1.tVector[j, i]
        return result

    def normSquared(self):
        return self.InnerProduct(self)

    def norm(self):
        return np.sqrt(self.normSquared())

    def ScalarMultiply(self, t):
        tVector_t = kendall3D_tVec(self.nPt)

        for i in range(self.nPt):
            for j in range(3):
                tVector_t.tVector[j, i] = self.tVector[j, i] * t

        return tVector_t

    def Write(self, filePath):
        infoList = [self.Type, self.nDim, self.nPt, self.tVector, False]

        with open(filePath, 'wb') as fp:
            pickle.dump(infoList, fp)

    def Read(self, filePath):
        with open(filePath, 'rb') as fp:
            infoList = pickle.load(fp)

        self.Type = infoList[0]
        self.nDim = infoList[1]
        self.nPt = infoList[2]
        self.tVector = infoList[3]


class kendall3D(object):
    def __init__(self, nPt):
        self.Type = "Kendall3D"
        self.nPt = nPt
        self.nDim = nPt - 2

        pt_base = np.zeros([3, nPt])
        pt_base[0, 0] = 1
        # pt_base[0, 1] = 0

        self.pt = pt_base

    def SetPoint(self, pt):
        if not pt.shape[1] == self.nPt:
            print("Error : # of Points does not match")
            return
        if not pt.shape[0] == 3:
            print("Error : Point should be 3D")
            return

        if not np.linalg.norm(pt) == 1:
            # print( "Warning : The point is not on a sphere")
            self.pt = np.asmatrix(pt)
            return

        self.pt = np.asmatrix(pt)

    def GetPoint(self):
        return self.pt

    def InnerProduct(self, ptA):
        result = 0
        for i in range(self.nPt):
            for j in range(3):
                result += self.pt[j, i] * ptA.pt[j, i]
        return result

    def normSquared(self):
        return self.InnerProduct(self)

    def norm(self):
        return np.sqrt(self.normSquared())

    def ExponentialMap(self, tVec):
        theta = tVec.norm()

        if theta < 1e-12:
            exp_pt = kendall3D(self.nPt)
            exp_pt.pt = self.pt
            return exp_pt

        if theta > np.pi * 2:
            theta = np.mod(theta, np.pi * 2)

        exp_pt = kendall3D(self.nPt)

        lhs = np.multiply(np.cos(theta), self.pt)
        rhs = np.multiply(np.sin(theta) / theta, tVec.tVector)

        exp_pt.pt = lhs + rhs
        exp_pt.pt = np.divide(exp_pt.pt, exp_pt.norm())

        return exp_pt

    def LogMap(self, another_pt):
        # m = np.matmul(self.pt, another_pt.pt.T)
        #
        # U, s, V = np.linalg.svd(m)
        #
        # rotation = np.matmul(U, V.T)

        qRot_pt = another_pt.pt  # np.matmul(rotation, another_pt.pt)

        qRot = kendall3D(self.nPt)
        qRot.SetPoint(qRot_pt)

        cosTheta = self.InnerProduct(qRot)

        tVec = kendall3D_tVec(self.nPt)

        tVec_mat = np.subtract(qRot.pt, np.multiply(cosTheta, self.pt))
        tVec.SetTangentVector(tVec_mat)

        length = tVec.norm()

        if length < 1e-12 or cosTheta >= 1.0 or cosTheta <= -1.0:
            tVec = kendall3D_tVec(self.nPt)
            return tVec

        tVec = tVec.ScalarMultiply(np.arccos(cosTheta) / length)

        return tVec

    def ParallelTranslate(self, v, w):
        # Modified version of parallel transport assuming unitP == unitJV
        vNorm = v.norm()
        pNorm = self.norm()

        if (vNorm < 1.0e-12 or pNorm < 1.0e-12):
            # print( "tVector too small" )
            return w

        unitV = v.ScalarMultiply(1.0 / vNorm)
        unitP = kendall3D_tVec(self.nPt)
        unitP.SetTangentVector(np.divide(self.pt, pNorm))

        wDotUnitV = unitV.InnerProduct(w)
        wDotUnitP = unitP.InnerProduct(w)  # should be 0 if w is a tangent vector at p

        # Component of w orthogonal to v and jv
        parallel_mat = np.add(np.multiply(wDotUnitV, unitV.tVector), np.multiply(wDotUnitP, unitP.tVector))
        orth_mat = np.subtract(w.tVector, parallel_mat)

        # Compute Parallel Translated V
        parallelUnitV_mat = np.add(np.multiply(unitP.tVector, -np.sin(vNorm)), np.multiply(np.cos(vNorm), unitV.tVector))

        # Compute Parallel Translated jV
        parallelUnitP_mat = np.add(np.multiply(np.cos(vNorm), unitP.tVector), np.multiply(np.sin(vNorm), unitV.tVector))

        # Add parallel translated v to orth, and get parallel translated w
        parallelW_paraV = np.add(np.multiply(wDotUnitV, parallelUnitV_mat), np.multiply(wDotUnitP, parallelUnitP_mat))
        parallelW_mat = np.add(parallelW_paraV, orth_mat)

        wParallelTranslated = kendall3D_tVec(self.nPt)
        wParallelTranslated.SetTangentVector(parallelW_mat)

        return wParallelTranslated

    def ParallelTranslateAtoB(self, a, b, w):
        v = a.LogMap(b)
        return a.ParallelTranslate(v, w)

    def ParallelTranslateToA(self, a, w):
        v = self.LogMap(a)
        return self.ParallelTranslate(v, w)

    def ParallelTranslateAtoBCurve(self, a, b, w: kendall3D_tVec, va: kendall3D_tVec, vb: kendall3D_tVec):
        # Generalized parallel transform of w from a to b along curve geodesics (eg: polynomials and splines)
        # va and vb are the pre/post translation tangent vectors of the curve at a and b respectively

        v = a.LogMap(b)
        va_b = a.ParallelTranslate(v, va)
        w_b = a.ParallelTranslate(v, w)

        unit_vb = kendall3D(self.nPt)
        unit_vb.SetPoint(vb.GetTangentVector())
        unit_va_b = kendall3D(self.nPt)
        unit_va_b.SetPoint(va_b.GetTangentVector())
        w_parallel_translated = unit_va_b.ParallelTranslateAtoB(unit_va_b, unit_vb, w_b)

        return w_parallel_translated

    def ScalarMultiply(self, t):
        p_t = kendall3D(self.nPt)

        for i in range(self.nPt):
            for j in range(3):
                p_t.pt[j, i] = self.pt[j, i] * t

        return p_t

    def GradientJacobi(self, v, J, dJ):
        vNorm = v.norm()

        if (vNorm < 1.0e-12):
            for i in range(self.nPt):
                for k in range(3):
                    J.tVector[k][i] = J.tVector[k][i] + dJ.tVector[k][i]

            return J

        VdotJ = v.InnerProduct(J)
        VdotJPrime = v.InnerProduct(dJ)

        scaleFactorJ = VdotJ / (vNorm * vNorm)
        scaleFactorJPrime = VdotJPrime / (vNorm * vNorm)

        jTang_mat = np.multiply(v.tVector, scaleFactorJ)
        jTang = kendall3D_tVec(self.nPt)
        jTang.SetTangentVector(jTang_mat)

        dJTang_mat = np.multiply(v.tVector, scaleFactorJPrime)
        dJTang = kendall3D_tVec(self.nPt)
        dJTang.SetTangentVector(dJTang_mat)

        jOrth_mat = np.subtract(J.tVector, jTang_mat)
        jOrth = kendall3D_tVec(self.nPt)
        jOrth.SetTangentVector(jOrth_mat)

        dJOrth_mat = np.subtract(dJ.tVector, dJTang_mat)
        dJOrth = kendall3D_tVec(self.nPt)
        dJOrth.SetTangentVector(dJOrth_mat)

        # A guess
        skew = np.zeros([3, 3])
        skew[0, 1] = -1
        skew[1, 0] = 1
        skew[0, 2] = 1
        skew[2, 0] = -1
        skew[1, 2] = -1
        skew[2, 1] = 1

        unitV = v.ScalarMultiply(1.0 / vNorm)

        w_mat = np.matmul(skew, unitV.tVector)
        w = kendall3D_tVec(self.nPt)
        w.SetTangentVector(w_mat)

        # Curvature 4 component
        jOrth4 = w.ScalarMultiply(w.InnerProduct(jOrth))
        dJOrth4 = w.ScalarMultiply(w.InnerProduct(dJOrth))

        # Curvature 1 Component
        jOrth1 = kendall3D_tVec(self.nPt)
        jOrth1.SetTangentVector(np.subtract(jOrth.tVector, jOrth4.tVector))

        dJOrth1 = kendall3D_tVec(self.nPt)
        dJOrth1.SetTangentVector(np.subtract(dJOrth.tVector, dJOrth4.tVector))

        # Orthogonal Parts
        jOrth.SetTangentVector(
            np.add(np.multiply(cos(vNorm), jOrth1.tVector), np.multiply(cos(2.0 * vNorm), jOrth4.tVector)))

        dJOrth.SetTangentVector(np.add(np.multiply(np.sin(vNorm) / vNorm, dJOrth1.tVector),
                                       np.multiply(0.5 * np.sin(2.0 * vNorm) / vNorm, dJOrth4.tVector)))

        J_dJ_mat = jTang.tVector + dJTang.tVector + jOrth.tVector + dJOrth.tVector
        J_dJ = kendall3D_tVec(self.nPt)
        J_dJ.SetTangentVector(J_DJ)

        J = self.ParallelTranslate(v, J_dJ)

        dJOrth_mat = jOrth1.ScalarMultiply(-vNorm * np.sin(vNorm)).tVector + jOrth4.ScalarMultiply(
            -2.0 * vNorm * sin(2.0 * vNorm)).tVector

        dJOrth.SetTangentVector(dJOrth_mat)

        ddJOrth_mat = dJOrth1.ScalarMultiply(cos(vNorm)).tVector + djOrth4.ScalarMultiply(cos(2.0 * vNorm)).tVector

        ddJOrth = kendall3D_tVec(self.nPt)
        ddJOrth.SetTangentVector(ddJOrth_mat)

        dJ_ddJ_mat = djTang.tVector + dJOrth.tVector + ddJOrth.tVector

        dJ_ddJ = kendall3D_tVec(self.nPt)

        dJ = self.ParallelTranslate(v, dJ_ddJ)

        return J, dJ

    def AdjointGradientJacobi(self, v, Jac, dJac):
        vNorm = v.norm()

        if (vNorm < 1.0e-12):
            for i in range(self.nPt):
                for j in range(3):
                    Jac.tVector[j][i] = Jac.tVector[j][i] + dJac.tVector[j][i]
            Jac_Updated = Jac
            dJac_Updated = dJac

            return Jac_Updated, dJac_Updated

        VdotJac = v.InnerProduct(Jac)
        VdotJacPrime = v.InnerProduct(dJac)

        scaleFactorJac = VdotJac / (vNorm * vNorm)
        scaleFactorJacPrime = VdotJacPrime / (vNorm * vNorm)

        jTang_mat = np.multiply(v.tVector, scaleFactorJac)
        jTang = kendall3D_tVec(self.nPt)
        jTang.SetTangentVector(jTang_mat)

        dJacTang_mat = np.multiply(v.tVector, scaleFactorJacPrime)
        dJacTang = kendall3D_tVec(self.nPt)
        dJacTang.SetTangentVector(dJacTang_mat)

        jOrth_mat = np.subtract(Jac.tVector, jTang_mat)
        jOrth = kendall3D_tVec(self.nPt)
        jOrth.SetTangentVector(jOrth_mat)

        dJacOrth_mat = np.subtract(dJac.tVector, dJacTang_mat)
        dJacOrth = kendall3D_tVec(self.nPt)
        dJacOrth.SetTangentVector(dJacOrth_mat)

        # A guess
        skew = np.zeros([3, 3])
        skew[0, 1] = -1
        skew[1, 0] = 1
        skew[0, 2] = 1
        skew[2, 0] = -1
        skew[1, 2] = -1
        skew[2, 1] = 1

        unitV = v.ScalarMultiply(1.0 / vNorm)

        w_mat = np.matmul(skew, unitV.tVector)
        w = kendall3D_tVec(self.nPt)
        w.SetTangentVector(w_mat)

        # Curvature 4 component
        jOrth4 = w.ScalarMultiply(w.InnerProduct(jOrth))
        dJacOrth4 = w.ScalarMultiply(w.InnerProduct(dJacOrth))

        # Curvature 1 Component
        jOrth1 = kendall3D_tVec(self.nPt)
        jOrth1.SetTangentVector(np.subtract(jOrth.tVector, jOrth4.tVector))

        dJacOrth1 = kendall3D_tVec(self.nPt)
        dJacOrth1.SetTangentVector(np.subtract(dJacOrth.tVector, dJacOrth4.tVector))

        # Orthogonal Parts
        jOrth.SetTangentVector(
            np.add(np.multiply(np.cos(vNorm), jOrth1.tVector), np.multiply(np.cos(2.0 * vNorm), jOrth4.tVector)))

        dJacOrth.SetTangentVector(np.add(np.multiply(-vNorm * np.sin(vNorm), dJacOrth1.tVector),
                                         np.multiply(-2.0 * vNorm * np.sin(2.0 * vNorm), dJacOrth4.tVector)))

        Jac_dJac_mat = jTang.tVector + jOrth.tVector + dJacOrth.tVector
        Jac_dJac = kendall3D_tVec(self.nPt)
        Jac_dJac.SetTangentVector(Jac_dJac_mat)

        Jac_Updated = self.ParallelTranslate(v, Jac_dJac)

        dJacOrth_mat = jOrth1.ScalarMultiply(np.sin(vNorm) / vNorm).tVector + jOrth4.ScalarMultiply(
            0.5 * np.sin(2.0 * vNorm) / vNorm).tVector

        dJacOrth.SetTangentVector(dJacOrth_mat)

        ddJacOrth_mat = dJacOrth1.ScalarMultiply(np.cos(vNorm)).tVector + dJacOrth4.ScalarMultiply(
            np.cos(2.0 * vNorm)).tVector

        ddJacOrth = kendall3D_tVec(self.nPt)
        ddJacOrth.SetTangentVector(ddJacOrth_mat)

        dJac_ddJac_mat = jTang.tVector + dJacTang.tVector + dJacOrth.tVector + ddJacOrth.tVector

        dJac_ddJac = kendall3D_tVec(self.nPt)
        dJac_Updated = self.ParallelTranslate(v, dJac_ddJac)

        return Jac_Updated, dJac_Updated

    def CurvatureTensor(self, x, y, z):
        # A guess
        skew = np.zeros([3, 3])
        skew[0, 1] = -1
        skew[1, 0] = 1
        skew[0, 2] = 1
        skew[2, 0] = -1
        skew[1, 2] = -1
        skew[2, 1] = 1

        jX_mat = np.matmul(skew, x.tVector)
        jY_mat = np.matmul(skew, y.tVector)
        jZ_mat = np.matmul(skew, z.tVector)

        jX = kendall3D_tVec(self.nPt)
        jX.SetTangentVector(jX_mat)

        jY = kendall3D_tVec(self.nPt)
        jY.SetTangentVector(jY_mat)

        jZ = kendall3D_tVec(self.nPt)
        jZ.SetTangentVector(jZ_mat)

        zxy_mat = np.multiply(self.InnerProduct(z, jX), jY)
        xyz_mat = np.multiply(self.InnerProduct(x, jY), jZ)
        yzx_mat = np.multiply(self.InnerProduct(y, jZ), jX)

        kCurv_mat = zxy_mat + yzx_mat - (np.multiply(2.0, xyz_mat))

        zDotX = self.InnerProduct(z, x)
        zDotY = self.InnerProduct(z, y)

        sphereCurv_mat = np.multiply(zDotX, y.tVector) - np.multiply(zDotY, x.tVector)

        curv_mat = kCurv_mat + sphereCurv_mat

        curv = kendall3D_tVec(self.nPt)
        curv.SetTangentVector(curv_mat)

        return curv

    def SectionalCurvature(self, x, y, z):
        curv = self.CurvatureTensor(x, y, z)

        sec = self.InnerProduct(curv, y)
        xx = self.normSquared(x)
        yy = self.normSquared(y)
        xy = self.InnerProduct(x, y)

        secCurv = sec / ((xx * yy) - (xy ** 2))
        return secCurv

    def ProjectTangent(self, p, v):
        meanV = np.average(v.tVector, axis=1)

        print(meanV.shape)

        hV = v

        for i in range(self.nPt):
            for j in range(3):
                hV.tVector[j, i] = v.tVector[j, i] - meanV[j]

        # A guess
        skew = np.zeros([3, 3])
        skew[0, 1] = -1
        skew[1, 0] = 1
        skew[0, 2] = 1
        skew[2, 0] = -1
        skew[1, 2] = -1
        skew[2, 1] = 1

        vert_mat = np.matmul(skew, p.pt)
        vert = kendall3D_tVec(self.nPt)
        vert.SetTangentVector(vert_mat)

        new_hV_mat = hV.tVector - np.multiply(p.InnerProduct(hV, vert), vert.tVector)

        hV.SetTangentVector(new_hV_mat)

        return hV

    def Write(self, filePath):
        infoList = [self.Type, self.nDim, self.nPt, self.pt, False]

        with open(filePath, 'wb') as fp:
            pickle.dump(infoList, fp)

    def Read(self, filePath):
        with open(filePath, 'rb') as fp:
            infoList = pickle.load(fp)

        self.Type = infoList[0]
        self.nDim = infoList[1]
        self.nPt = infoList[2]
        self.pt = infoList[3]
