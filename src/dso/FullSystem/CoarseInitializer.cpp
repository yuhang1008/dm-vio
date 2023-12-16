/**
* This file is part of DSO, written by Jakob Engel.
* It has been modified by Lukas von Stumberg for the inclusion in DM-VIO (http://vision.in.tum.de/dm-vio).
*
* Copyright 2022 Lukas von Stumberg <lukas dot stumberg at tum dot de>
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/CoarseInitializer.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "util/nanoflann.h"

#include <opencv2/highgui/highgui.hpp>

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{

CoarseInitializer::CoarseInitializer(int ww, int hh)
        : thisToNext_aff(0, 0), thisToNext(SE3())
{
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		points[lvl] = 0;
		numPoints[lvl] = 0;
	}

	// for what?
	JbBuffer = new Vec10f[ww*hh]; // 10x1 x (wxh) ??
	JbBuffer_new = new Vec10f[ww*hh];


	frameID=-1;
	fixAffine=true;
	printDebug=false;

	// Eigen::DiagonalMatrix<float, 8> wM;
	// covariance matrix? rot:3 translation:3 scale:?2 a,b?
	wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_ROT;
	wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_TRANS;
	wM.diagonal()[6] = SCALE_A;
	wM.diagonal()[7] = SCALE_B;
}
CoarseInitializer::~CoarseInitializer()
{
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		if(points[lvl] != 0) delete[] points[lvl];
	}

	delete[] JbBuffer;
	delete[] JbBuffer_new;
}


bool CoarseInitializer::trackFrame(FrameHessian *newFrameHessian, std::vector<IOWrap::Output3DWrapper*> &wraps)
{
	newFrame = newFrameHessian;

	// why need so many wrappers? why need to push into every wrapper?
    for(IOWrap::Output3DWrapper* ow : wraps)
        ow->pushLiveFrame(newFrameHessian); 

	int maxIterations[] = {5,5,10,30,50}; //?

	alphaK = 2.5*2.5;//*freeDebugParam1*freeDebugParam1;
	alphaW = 150*150;//*freeDebugParam2*freeDebugParam2;
	regWeight = 0.8;//*freeDebugParam4;
	couplingWeight = 1;//*freeDebugParam5;

	// initialize some parameters
	if(!snapped) // snapped set false during set first frame
	{
		thisToNext.translation().setZero();
		for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
		{
			int npts = numPoints[lvl];
			Pnt* ptsl = points[lvl];
			for(int i=0;i<npts;i++)
			{
				ptsl[i].iR = 1; //?
				ptsl[i].idepth_new = 1; //?
				ptsl[i].lastHessian = 0; //?
			}
		}
	}

	// reference
	SE3 refToNew_current = thisToNext; // constant motion model?

	//?
	Vec3f latestRes = Vec3f::Zero();

	// reference affine function
	AffLight refToNew_aff_current = thisToNext_aff; // initialized as a=b=0 in first frame

	// both have exposure times
	if(firstFrame->ab_exposure>0 && newFrame->ab_exposure>0)
		// coarse approximation of current affine function
		// a related to exposure and b related to error?
		// a is propotional to log t, see eq 4
		refToNew_aff_current = AffLight(logf(newFrame->ab_exposure /  firstFrame->ab_exposure),0); // ln(tn/to), 0

	// coarse to fine (top to bottom)
	for(int lvl=pyrLevelsUsed-1; lvl>=0; lvl--)
	{
		// propagate except for coarest level
		if(lvl<pyrLevelsUsed-1)
			// propagate the inverse depth of pixel from coarse to fine, (above to current)
			// make the depth estimation for each pixel more accurate
			propagateDown(lvl+1);

		Mat88f H,Hsc; Vec8f b,bsc;

		// at this level:
		// set point energy to 0
		// set idepth_new to current idepth
		// if is the coarest level, for the bad points, try to initialize depth using nn points
		resetPoints(lvl);

		Vec3f resOld = calcResAndGS(lvl, H, b, Hsc, bsc, refToNew_current, refToNew_aff_current, false);

		applyStep(lvl);

		float lambda = 0.1;
		float eps = 1e-4;
		int fails=0;

		if(printDebug)
		{
			printf("lvl %d, it %d (l=%f) %s: %.3f+%.5f -> %.3f+%.5f (%.3f->%.3f) (|inc| = %f)! \t",
					lvl, 0, lambda,
					"INITIA",
					sqrtf((float)(resOld[0] / resOld[2])),
					sqrtf((float)(resOld[1] / resOld[2])),
					sqrtf((float)(resOld[0] / resOld[2])),
					sqrtf((float)(resOld[1] / resOld[2])),
					(resOld[0]+resOld[1]) / resOld[2],
					(resOld[0]+resOld[1]) / resOld[2],
					0.0f);
			std::cout << refToNew_current.log().transpose() << " AFF " << refToNew_aff_current.vec().transpose() <<"\n";
		}

		int iteration=0;
		while(true)
		{
			Mat88f Hl = H;
			for(int i=0;i<8;i++) Hl(i,i) *= (1+lambda);
			Hl -= Hsc*(1/(1+lambda));
			Vec8f bl = b - bsc*(1/(1+lambda));

			Hl = wM * Hl * wM * (0.01f/(w[lvl]*h[lvl]));
			bl = wM * bl * (0.01f/(w[lvl]*h[lvl]));


            Vec8f inc;
            SE3 refToNew_new;
            if (fixAffine)
            {
                // Note as we set the weights of rotation and translation to 1 the wM is just the identity in this case.
                inc.head<6>() = -(wM.toDenseMatrix().topLeftCorner<6, 6>() *
                                  (Hl.topLeftCorner<6, 6>().ldlt().solve(bl.head<6>())));
                inc.tail<2>().setZero();
            } else
                inc = -(wM * (Hl.ldlt().solve(bl)));    //=-H^-1 * b.

            double incNorm = inc.norm();

            refToNew_new = SE3::exp(inc.head<6>().cast<double>()) * refToNew_current;

			AffLight refToNew_aff_new = refToNew_aff_current;
			refToNew_aff_new.a += inc[6];
			refToNew_aff_new.b += inc[7];
			doStep(lvl, lambda, inc);


			Mat88f H_new, Hsc_new; Vec8f b_new, bsc_new;
			Vec3f resNew = calcResAndGS(lvl, H_new, b_new, Hsc_new, bsc_new, refToNew_new, refToNew_aff_new, false);
			Vec3f regEnergy = calcEC(lvl);

			float eTotalNew = (resNew[0]+resNew[1]+regEnergy[1]);
			float eTotalOld = (resOld[0]+resOld[1]+regEnergy[0]);


			bool accept = eTotalOld > eTotalNew;

			if(printDebug)
			{
				printf("lvl %d, it %d (l=%f) %s: %.5f + %.5f + %.5f -> %.5f + %.5f + %.5f (%.2f->%.2f) (|inc| = %f)! \t",
						lvl, iteration, lambda,
						(accept ? "ACCEPT" : "REJECT"),
						sqrtf((float)(resOld[0] / resOld[2])),
						sqrtf((float)(regEnergy[0] / regEnergy[2])),
						sqrtf((float)(resOld[1] / resOld[2])),
						sqrtf((float)(resNew[0] / resNew[2])),
						sqrtf((float)(regEnergy[1] / regEnergy[2])),
						sqrtf((float)(resNew[1] / resNew[2])),
						eTotalOld / resNew[2],
						eTotalNew / resNew[2],
						incNorm);
				std::cout << refToNew_new.log().transpose() << " AFF " << refToNew_aff_new.vec().transpose() <<"\n";
			}

			if(accept)
			{
				if(resNew[1] == alphaK*numPoints[lvl])
					snapped = true;
				H = H_new;
				b = b_new;
				Hsc = Hsc_new;
				bsc = bsc_new;
				resOld = resNew;
				refToNew_aff_current = refToNew_aff_new;
				refToNew_current = refToNew_new;
				applyStep(lvl);
				optReg(lvl);
				lambda *= 0.5;
				fails=0;
				if(lambda < 0.0001) lambda = 0.0001;
			}
			else
			{
				fails++;
				lambda *= 4;
				if(lambda > 10000) lambda = 10000;
			}

			bool quitOpt = false;

			if(!(incNorm > eps) || iteration >= maxIterations[lvl] || fails >= 2)
			{
				Mat88f H,Hsc; Vec8f b,bsc;

				quitOpt = true;
			}


			if(quitOpt) break;
			iteration++;
		}
		latestRes = resOld;

	}

	thisToNext = refToNew_current;
	thisToNext_aff = refToNew_aff_current;

	for(int i=0;i<pyrLevelsUsed-1;i++)
		propagateUp(i);

	frameID++;
	if(!snapped) snappedAt=0;

	if(snapped && snappedAt==0)
		snappedAt = frameID;

    debugPlot(0,wraps);

	return snapped && frameID > snappedAt+5;
}

void CoarseInitializer::debugPlot(int lvl, std::vector<IOWrap::Output3DWrapper*> &wraps)
{
    bool needCall = false;
    for(IOWrap::Output3DWrapper* ow : wraps)
        needCall = needCall || ow->needPushDepthImage();
    if(!needCall) return;


	int wl = w[lvl], hl = h[lvl];
	Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];

	MinimalImageB3 iRImg(wl,hl);

	for(int i=0;i<wl*hl;i++)
		iRImg.at(i) = Vec3b(colorRef[i][0],colorRef[i][0],colorRef[i][0]);


	int npts = numPoints[lvl];

	float nid = 0, sid=0;
	for(int i=0;i<npts;i++)
	{
		Pnt* point = points[lvl]+i;
		if(point->isGood)
		{
			nid++;
			sid += point->iR;
		}
	}
	float fac = nid / sid;


	for(int i=0;i<npts;i++)
	{
		Pnt* point = points[lvl]+i;

		if(!point->isGood)
			iRImg.setPixel9(point->u+0.5f,point->v+0.5f,Vec3b(0,0,0));

		else
			iRImg.setPixel9(point->u+0.5f,point->v+0.5f,makeRainbow3B(point->iR*fac));
	}


	//IOWrap::displayImage("idepth-R", &iRImg, false);
    for(IOWrap::Output3DWrapper* ow : wraps)
        ow->pushDepthImage(&iRImg);
}



// calculates residual, Hessian and Hessian-block neede for re-substituting depth.

// lvl: Level of image pyramid.
// H_out, H_out_sc: Output Hessians (probably 8x8 matrices).
// b_out, b_out_sc: Output vectors associated with Hessians.
// refToNew: Transformation (rotation and translation) from reference to new frame.
// refToNew_aff: Affine light transformation between frames.
Vec3f CoarseInitializer::calcResAndGS(
		int lvl, Mat88f &H_out, Vec8f &b_out,
		Mat88f &H_out_sc, Vec8f &b_out_sc,
		const SE3 &refToNew, AffLight refToNew_aff,
		bool plot)
{
	int wl = w[lvl], hl = h[lvl];

	// frame to frame
	// dIp saves (I dx dy)
	Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];
	Eigen::Vector3f* colorNew = newFrame->dIp[lvl];


	Mat33f RKi = (refToNew.rotationMatrix() * Ki[lvl]).cast<float>(); // rotation * intrinsic matrix
	Vec3f t = refToNew.translation().cast<float>(); //translation
	Eigen::Vector2f r2new_aff = Eigen::Vector2f(exp(refToNew_aff.a), refToNew_aff.b);

	float fxl = fx[lvl];
	float fyl = fy[lvl];
	float cxl = cx[lvl];
	float cyl = cy[lvl];

	// initialize these for what
    for(auto&& acc9 : acc9s)
    {
        acc9.initialize();
    }
    for(auto&& E : accE)
    {
        E.initialize();
    }

	int npts = numPoints[lvl]; //n_effective point at this level
	Pnt* ptsl = points[lvl]; //pts last frame

    // This part takes most of the time for this method --> parallelize this only.

	// lambda function, [&] is to capture local variables 
	// reduce.reduce(processPointsForReduce, 0, npts, 50);
    auto processPointsForReduce = [&](int min=0, int max=1, double* stats=0, int tid=0)
    {	

        auto& acc9 = acc9s[tid]; //  Mat99f H; Vec9f b; size_t num;
        auto& E = accE[tid]; // save energy?

        for(int i = min; i < max; i++) // for specific points
        {
            Pnt* point = ptsl + i;

            point->maxstep = 1e10;

			// if point is not good
			// add point energy to accumulaor, point->isGood_new = false
            if(!point->isGood)
            {
                E.updateSingle((float) (point->energy[0]));
                point->energy_new = point->energy;
                point->isGood_new = false;
                continue;
            }

			// else (point->isGood)
			// typedef Eigen::Matrix<float,8,1> VecNRf; residules?
            VecNRf dp0;
            VecNRf dp1;
            VecNRf dp2;
            VecNRf dp3;
            VecNRf dp4;
            VecNRf dp5;
            VecNRf dp6;
            VecNRf dp7;

            VecNRf dd;
            VecNRf r;

			//JbBuffer_new = new Vec10f[ww*hh];
			// every point has 10 vector positions
			// 0-7: sum(dd * dp). 8: sum(res*dd). 9: 1/(1+sum(dd*dd))=inverse hessian entry.
            JbBuffer_new[i].setZero(); //

            // sum over all residuals from pattern pixels.
            bool isGood = true;
            float energy = 0; // total energy of residual of the pattern
            for(int idx = 0; idx < patternNum; idx++) // idx: pattern index
            {	
                int dx = patternP[idx][0];
                int dy = patternP[idx][1];

				// pattern point projecte to current frame (constant moving model?)
				// check the issue ============== problem ========================
				// important pt here is 1/z_old * point_(old point in new frame)
                Vec3f pt = RKi * Vec3f(point->u + dx, point->v + dy, 1) + t * point->idepth_new;

				// 2d coordinates of projected point at nomalized plane
                float u = pt[0] / pt[2];
                float v = pt[1] / pt[2];
				// pixel coordinates
                float Ku = fxl * u + cxl;
                float Kv = fyl * v + cyl;
                float new_idepth = point->idepth_new / pt[2]; // (1/z_i) / (z_i/z_j) = 1 / z_n
				
				// only consider pattern points in image range and has positive inverse depth
                if(!(Ku > 1 && Kv > 1 && Ku < wl - 2 && Kv < hl - 2 && new_idepth > 0))
                {
                    isGood = false;
                    break;
                }

				// use transformed point pixel position (float)
				// hit color is at pixel (Ku, Kv), the interpulated dip from new frame
                Vec3f hitColor = getInterpolatedElement33(colorNew, Ku, Kv, wl);
                //Vec3f hitColor = getInterpolatedElement33BiCub(colorNew, Ku, Kv, wl);

                //float rlR = colorRef[point->u+dx + (point->v+dy) * wl][0];


				// point of last frame interpulated I on last frame
                float rlR = getInterpolatedElement31(colorRef, point->u + dx, point->v + dy, wl);

				// both I of new and old frame cannot be infinite
                if(!std::isfinite(rlR) || !std::isfinite((float) hitColor[0]))
                {
                    isGood = false;
                    break;
                }

                float residual = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1]; // eq(4), see notes also


                float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual); // Huber funtion coefficient
                energy += hw * residual * residual * (2 - hw);
				
				//pt 2 = z_n/z_o
                float dxdd = (t[0] - t[2] * u) / pt[2];
                float dydd = (t[1] - t[2] * v) / pt[2];

                if(hw < 1) hw = sqrtf(hw); // else hw = 1
                float dxInterp = hw * hitColor[1] * fxl; // hw*dx*fx
                float dyInterp = hw * hitColor[2] * fyl; // hw*dy*fy

				// eq 29 corresponding rows, dr/ d_pose
                dp0[idx] = new_idepth * dxInterp; // ro*hw*dx*fx
                dp1[idx] = new_idepth * dyInterp; // ro*hw*dy*fy
                dp2[idx] = -new_idepth * (u * dxInterp + v * dyInterp); // -ro * (uj*hw*dx*fx + vj*hw*dy*fy)
                dp3[idx] = -u * v * dxInterp - (1 + v * v) * dyInterp; // -uj*vj*hw*dx*fx - (1+vj*vj)*hw*dy*fy
                dp4[idx] = (1 + u * u) * dxInterp + u * v * dyInterp; // (1+uj*uj)*hw*dx*fx + u*v*hw*dy*fy
                dp5[idx] = -v * dxInterp + u * dyInterp; // -vj*hw*dx*fx + uj*hw*dy*fy

				// eq 14 and 15 dr/ d affine coef
                dp6[idx] = -hw * r2new_aff[0] * rlR; // -hw * a_what * Ii  
                dp7[idx] = -hw * 1; // -hw

				//eq 24 in notes, d_r/d_inverseDepth
                dd[idx] = dxInterp * dxdd + dyInterp * dydd;

				// eq 10, actual lose
                r[idx] = hw * residual;

				//pt 2 = z_j/zi, see eq 23, inverse norm of dpixel/d_inverse_depth ?
                float maxstep = 1.0f / Vec2f(dxdd * fxl, dydd * fyl).norm();
                if(maxstep < point->maxstep) point->maxstep = maxstep; // minimum max_depth

                // immediately compute dp*dd' and dd*dd' in JbBuffer_new.
				// note i is the index of point from a certain thread
                JbBuffer_new[i][0] += dp0[idx] * dd[idx];
                JbBuffer_new[i][1] += dp1[idx] * dd[idx];
                JbBuffer_new[i][2] += dp2[idx] * dd[idx];
                JbBuffer_new[i][3] += dp3[idx] * dd[idx];
                JbBuffer_new[i][4] += dp4[idx] * dd[idx];
                JbBuffer_new[i][5] += dp5[idx] * dd[idx];

				// d_affine * d_d
                JbBuffer_new[i][6] += dp6[idx] * dd[idx];
                JbBuffer_new[i][7] += dp7[idx] * dd[idx];

                JbBuffer_new[i][8] += r[idx] * dd[idx];
                JbBuffer_new[i][9] += dd[idx] * dd[idx];
            }

			// !isGood means out of image range or I of (last or curren) image is infinite
			//  energy is the total huber loss of residual
            if(!isGood || energy > point->outlierTH * 20)
            {
                E.updateSingle((float) (point->energy[0])); //update last energy
                point->isGood_new = false;
                point->energy_new = point->energy; // new_energy = last_energy
                continue;
            }


            // add into energy.
            E.updateSingle(energy);
            point->isGood_new = true;
            point->energy_new[0] = energy; // energy of total surrounding parterns 

            // update Hessian matrix.
			// i = 0; i < 11 ; 1 += 4
			// i = 0, 4 ???  patternNum = 8
			// 0-5 pose, 6 and 7 affine, r residual

			// follow order, from left to right, from top to bottom
			// x1x1, x1x2, x1x3, x1x4, x1x5, x1x6, x1a, x1b, x1r,
			// 		 x2x2, x2x3, x2x4, x2x5, x2x6, x2a, x2b, x2r,
			// 			   x3x3, x3x4, x3x5, x3x6, x3a, x3b, x3r,
			// 					 x4x4, x4x5, x4x6, x4a, x4b, x4r,
			// 						   x5x5, x5x6, x5a, x5b, x5r,
			// 								 x6x6, x6a, x6b, x6r,
			// 									   aa,  ab,  ar,
			// 					 						bb,  br,
			// 												 rr.

			// so i will be 0 and 4, each time update 4 values
            for(int i = 0; i + 3 < patternNum; i += 4)
                acc9.updateSSE(
                        _mm_load_ps(((float*) (&dp0)) + i),
                        _mm_load_ps(((float*) (&dp1)) + i),
                        _mm_load_ps(((float*) (&dp2)) + i),
                        _mm_load_ps(((float*) (&dp3)) + i),
                        _mm_load_ps(((float*) (&dp4)) + i),
                        _mm_load_ps(((float*) (&dp5)) + i),
                        _mm_load_ps(((float*) (&dp6)) + i),
                        _mm_load_ps(((float*) (&dp7)) + i),
                        _mm_load_ps(((float*) (&r)) + i));

			// for handling partternNum is not multiple of 4, add the rest 1 by 1
            for(int i = ((patternNum >> 2) << 2); i < patternNum; i++)
                acc9.updateSingle( //updare at the first elements for 45 entries
                        (float) dp0[i], (float) dp1[i], (float) dp2[i], (float) dp3[i],
                        (float) dp4[i], (float) dp5[i], (float) dp6[i], (float) dp7[i],
                        (float) r[i]);
        }
    };


    reduce.reduce(processPointsForReduce, 0, npts, 50); // first, end, step size

    for(auto&& acc9 : acc9s) //class Accumulator9, stores H and b
    {
        acc9.finish();
    }
    for(auto&& E : accE) // class Accumulator11, stores error
    {
        E.finish();
    }


	// calculate alpha energy, and decide if we cap it.
	Accumulator11 EAlpha;
	EAlpha.initialize();
	for(int i=0;i<npts;i++) // pts from last frame
	{
		Pnt* point = ptsl+i;
		if(!point->isGood_new)
		{
            // This should actually be EAlpha, but it seems like fixing this might change the optimal values of some
            // parameters, so it's kept like it is (see https://github.com/JakobEngel/dso/issues/52)
            // At the moment, this code will not change the value of E.A (because E.finish() is not called again after
            // this. It will however change E.num.
			accE[0].updateSingle((float)(point->energy[1]));
		}
		else
		{	
			// far -> small inverse_depth -> bigger energy_new[1] // if far add more energu since it may be uncertain?
			// close -> big inverse depth -> smaller energy_new[1] // else if close, energy is accurate, add small energy
			// add to accE[0]
			point->energy_new[1] = (point->idepth_new-1)*(point->idepth_new-1);
			accE[0].updateSingle((float)(point->energy_new[1]));
		}
	}
	EAlpha.finish(); //empty

	// alphaEnergy = factor*(translation.norm()* npts)
	// larger the translation, more points, higher the alphaEnergy
	float alphaEnergy = alphaW*(EAlpha.A + refToNew.translation().squaredNorm() * npts);

	// compute alpha opt.
	float alphaOpt;
	if(alphaEnergy > alphaK*npts)
	{
		alphaOpt = 0;
		alphaEnergy = alphaK*npts;
	}
	else
	{
		alphaOpt = alphaW; // 150*150
	}

	acc9SC.initialize();
	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		if(!point->isGood_new)
			continue;

		// if good points
		// JbBuffer_new[i][9] += dd[idx] * dd[idx];
		// bigger the gradient of idepth, higher the Hessian
		point->lastHessian_new = JbBuffer_new[i][9];
		
		// JbBuffer_new[i][8] += r[idx] * dd[idx];
        // JbBuffer_new[i][9] += dd[idx] * dd[idx];

		// if alphaOpt != 0, which means alphaEnergy is small, 
		// which means translation.norm()* npts is small
		JbBuffer_new[i][8] += alphaOpt*(point->idepth_new - 1);
		JbBuffer_new[i][9] += alphaOpt;

		// alphaOpt==0, which means translation.norm()* npts is big
		if(alphaOpt==0)
		{	
			JbBuffer_new[i][8] += couplingWeight*(point->idepth_new - point->iR);
			JbBuffer_new[i][9] += couplingWeight;
		}

		JbBuffer_new[i][9] = 1/(1+JbBuffer_new[i][9]);
		acc9SC.updateSingleWeighted(
				(float)JbBuffer_new[i][0],(float)JbBuffer_new[i][1],(float)JbBuffer_new[i][2],(float)JbBuffer_new[i][3],
				(float)JbBuffer_new[i][4],(float)JbBuffer_new[i][5],(float)JbBuffer_new[i][6],(float)JbBuffer_new[i][7],
				(float)JbBuffer_new[i][8],(float)JbBuffer_new[i][9]);
	}
	acc9SC.finish();


    H_out.setZero();
    b_out.setZero();
    // This needs to sum up the acc9s from all the workers!
    for(auto&& acc9 : acc9s)
    {
        H_out += acc9.H.topLeftCorner<8,8>();// / acc9.num;
        b_out += acc9.H.topRightCorner<8,1>();// / acc9.num;
    }
	H_out_sc = acc9SC.H.topLeftCorner<8,8>();// / acc9.num;
	b_out_sc = acc9SC.H.topRightCorner<8,1>();// / acc9.num;



	H_out(0,0) += alphaOpt*npts;
	H_out(1,1) += alphaOpt*npts;
	H_out(2,2) += alphaOpt*npts;

	Vec3f tlog = refToNew.log().head<3>().cast<float>();
	b_out[0] += tlog[0]*alphaOpt*npts;
	b_out[1] += tlog[1]*alphaOpt*npts;
	b_out[2] += tlog[2]*alphaOpt*npts;


	// Add zero prior to translation.
    // setting_weightZeroPriorDSOInitY is the squared weight of the prior residual.
    H_out(1, 1) += setting_weightZeroPriorDSOInitY;
    b_out(1) += setting_weightZeroPriorDSOInitY * refToNew.translation().y();

    H_out(0, 0) += setting_weightZeroPriorDSOInitX;
    b_out(0) += setting_weightZeroPriorDSOInitX * refToNew.translation().x();

    double A = 0;
    int num = 0;
    for(auto&& E : accE)
    {
        A += E.A;
        num += E.num;
    }

	return Vec3f(A, alphaEnergy, num);
}


float CoarseInitializer::rescale()
{
	float factor = 20*thisToNext.translation().norm();
//	float factori = 1.0f/factor;
//	float factori2 = factori*factori;
//
//	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
//	{
//		int npts = numPoints[lvl];
//		Pnt* ptsl = points[lvl];
//		for(int i=0;i<npts;i++)
//		{
//			ptsl[i].iR *= factor;
//			ptsl[i].idepth_new *= factor;
//			ptsl[i].lastHessian *= factori2;
//		}
//	}
//	thisToNext.translation() *= factori;

	return factor;
}


Vec3f CoarseInitializer::calcEC(int lvl)
{
	if(!snapped) return Vec3f(0,0,numPoints[lvl]);
	AccumulatorX<2> E;
	E.initialize();
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		Pnt* point = points[lvl]+i;
		if(!point->isGood_new) continue;
		float rOld = (point->idepth-point->iR);
		float rNew = (point->idepth_new-point->iR);
		E.updateNoWeight(Vec2f(rOld*rOld,rNew*rNew));

		//printf("%f %f %f!\n", point->idepth, point->idepth_new, point->iR);
	}
	E.finish();

	//printf("ER: %f %f %f!\n", couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], (float)E.num.numIn1m);
	return Vec3f(couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], E.num);
}

// use the nn points to regulate the inv-depth of each selected points,
// make the depth estimation more consistent and trustable
void CoarseInitializer::optReg(int lvl)
{
	int npts = numPoints[lvl];
	Pnt* ptsl = points[lvl];
	// must be snapped 
	// at first frame it is set 'false'
	// so if currently is the second frame, will do nothing
	if(!snapped) 
	{
		return;
	}


	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		if(!point->isGood) continue;

		float idnn[10];
		int nnn=0;
		for(int j=0;j<10;j++)
		{
			if(point->neighbours[j] == -1) continue;
			Pnt* other = ptsl+point->neighbours[j];
			if(!other->isGood) continue;
			idnn[nnn] = other->iR;
			nnn++;
		}

		if(nnn > 2)
		{
			std::nth_element(idnn,idnn+nnn/2,idnn+nnn);
			point->iR = (1-regWeight)*point->idepth + regWeight*idnn[nnn/2]; // here save the interpulated inverse depth
		}
	}

}



void CoarseInitializer::propagateUp(int srcLvl)
{
	assert(srcLvl+1<pyrLevelsUsed);
	// set idepth of target

	int nptss= numPoints[srcLvl];
	int nptst= numPoints[srcLvl+1];
	Pnt* ptss = points[srcLvl];
	Pnt* ptst = points[srcLvl+1];

	// set to zero.
	for(int i=0;i<nptst;i++)
	{
		Pnt* parent = ptst+i;
		parent->iR=0;
		parent->iRSumNum=0;
	}

	for(int i=0;i<nptss;i++)
	{
		Pnt* point = ptss+i;
		if(!point->isGood) continue;

		Pnt* parent = ptst + point->parent;
		parent->iR += point->iR * point->lastHessian;
		parent->iRSumNum += point->lastHessian;
	}

	for(int i=0;i<nptst;i++)
	{
		Pnt* parent = ptst+i;
		if(parent->iRSumNum > 0)
		{
			parent->idepth = parent->iR = (parent->iR / parent->iRSumNum);
			parent->isGood = true;
		}
	}

	optReg(srcLvl+1);
}

// propagate the inverse depth of pixel from srcLvl to level below it
// make the depth estimation for each pixel more accurate
void CoarseInitializer::propagateDown(int srcLvl)
{
	assert(srcLvl>0);
	// set idepth of target

	int nptst= numPoints[srcLvl-1]; 
	Pnt* ptss = points[srcLvl]; //src
	Pnt* ptst = points[srcLvl-1]; //to

	for(int i=0;i<nptst;i++) // for all pts in srclev-1
	{
		Pnt* point = ptst+i;
		Pnt* parent = ptss+point->parent;

		// if parent point is not good
		// Hessian value at each point might represent the confidence or reliability of the depth estimate
		if(!parent->isGood || parent->lastHessian < 0.1) continue;
		
		// If the current point itself is not "good":
    	// Directly inherit the depth (inverse depth iR in this context) from its parent.
    	// Mark the point as "good" and reset its Hessian value.
		if(!point->isGood)
		{
			point->iR = point->idepth = point->idepth_new = parent->iR;
			point->isGood=true;
			point->lastHessian=0; // last time not good
		}
		// it seems to be too tricky
		// both current pts and parent point are good
		// interpulation
		else
		{
			float newiR = (point->iR*point->lastHessian*2 + parent->iR*parent->lastHessian) / (point->lastHessian*2+parent->lastHessian);
			point->iR = point->idepth = point->idepth_new = newiR;
		}
	}

	// use the nn points to regulate the inv-depth of each selected points, save to iR
	// make the depth estimation more consistent and trustable
	optReg(srcLvl-1);
}


void CoarseInitializer::makeGradients(Eigen::Vector3f** data)
{
	for(int lvl=1; lvl<pyrLevelsUsed; lvl++)
	{
		int lvlm1 = lvl-1;
		int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

		Eigen::Vector3f* dINew_l = data[lvl];
		Eigen::Vector3f* dINew_lm = data[lvlm1];

		for(int y=0;y<hl;y++)
			for(int x=0;x<wl;x++)
				dINew_l[x + y*wl][0] = 0.25f * (dINew_lm[2*x   + 2*y*wlm1][0] +
													dINew_lm[2*x+1 + 2*y*wlm1][0] +
													dINew_lm[2*x   + 2*y*wlm1+wlm1][0] +
													dINew_lm[2*x+1 + 2*y*wlm1+wlm1][0]);

		for(int idx=wl;idx < wl*(hl-1);idx++)
		{
			dINew_l[idx][1] = 0.5f*(dINew_l[idx+1][0] - dINew_l[idx-1][0]);
			dINew_l[idx][2] = 0.5f*(dINew_l[idx+wl][0] - dINew_l[idx-wl][0]);
		}
	}
}

// at first frame:
// make calibration paras
// in all levels, select feature pixels based on irradiance and gradient change:
	// at the first level:
	// make image histogram, feature pixel selection based on irradiance in different levels
	// stores the result in map_out, which indicate which level the pixel is selected: enum PixelSelectorStatus {PIXSEL_VOID=0, PIXSEL_1, PIXSEL_2, PIXSEL_3};
	// return number of pixel that is selected

	// at other levels
	// statusMapB will save the It only marks which pixels are selected based on having the highest gradient magnitude 
	// in any of the considered directions (XX, YY, XY, YX) within their respective grid cells (size determined by pot).
	// return  number of selected pixels
	// will do this -coarse-to-fine recrusively	based on the density desired	

	// Pnt* pl = points[lvl]; for these initialized points 
	// use kdtree to find 10 nearest points and 
	// if not fiest level, save the parent and neighbors indices and distance
void CoarseInitializer::setFirst(CalibHessian* HCalib, FrameHessian* newFrameHessian)
{

	makeK(HCalib); // make calibration paras of different pyramid levels from HCalib
	firstFrame = newFrameHessian;

	// generate random pixel mask, saved in randomPattern, 
	// make and deterministic block dividing for each level, save the block w h 
	PixelSelector sel(w[0],h[0]); 
	
	float* statusMap = new float[w[0]*h[0]];
	bool* statusMapB = new bool[w[0]*h[0]];
	float densities[] = {0.03,0.05,0.15,0.5,1};

	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		sel.currentPotential = 3;
		int npts;
		if(lvl == 0)

// need to figure out the difference between two first level and others

		// at the first level:
		// make image histogram, feature pixel selection based on grascale value change on arbitary 16 directions in different levels
		// stores the result in map_out, which indicate which level the pixel is selected: enum PixelSelectorStatus {PIXSEL_VOID=0, PIXSEL_1, PIXSEL_2, PIXSEL_3};
		// return number of pixel that is selected
														// number of point wanted
			npts = sel.makeMaps(firstFrame, statusMap, densities[lvl]*w[0]*h[0], 1, false, 2); // return number of points actually selected
		else

		// at other levels
		// statusMapB (just bool) marks which pixels are selected based on having the highest gradient magnitude 
		// in any of the considered directions (XX, YY, XY, YX) within their respective grid cells (size determined by pot).
		// return  number of selected pixels
		// will do this recrusively	based on the density desired	
			npts = makePixelStatus(firstFrame->dIp[lvl], statusMapB, w[lvl], h[lvl], densities[lvl]*w[0]*h[0]);


		// Pnt* points[PYR_LEVELS];

		// ! the following will initialize all points that are selected in above 2 lines of codes

		if(points[lvl] != 0) delete[] points[lvl];
		points[lvl] = new Pnt[npts]; // initialize points obj of the level
		int wl = w[lvl], hl = h[lvl];
		Pnt* pl = points[lvl]; // points ptr of this level
		int nl = 0; //point index

		// initilize points (not pixel) of selected pixels, pass the u,v, type
		for(int y=patternPadding+1;y<hl-patternPadding-2;y++)
		for(int x=patternPadding+1;x<wl-patternPadding-2;x++)
		{
			//if(x==2) printf("y=%d!\n",y);
			//   other level & point selected      first level & point valid
			if((lvl!=0 && statusMapB[x+y*wl]) || (lvl==0 && statusMap[x+y*wl] != 0))
			{
				//assert(patternNum==9);
				// pixel position
				pl[nl].u = x+0.1;
				pl[nl].v = y+0.1;

				// init inverse depth !!! can modify
				pl[nl].idepth = 1;
				// ?
				pl[nl].iR = 1;
				pl[nl].isGood=true;

				// pixel energy? // Vec2f (UenergyPhotometric, energyRegularizer)
				// Hessian value at each point might represent the confidence or reliability of the depth estimate
				pl[nl].energy.setZero();
				pl[nl].lastHessian=0;
				pl[nl].lastHessian_new=0;
				
				// important
				// first level: actual selected level
				// other level: 1
				pl[nl].my_type= (lvl!=0) ? 1 : statusMap[x+y*wl]; 

				Eigen::Vector3f* cpt = firstFrame->dIp[lvl] + x + y*w[lvl]; // <grayscale, dx, dy >current point

				// calculate and accumulate gradient norm of each pixel inside a pattern arround the select pixel
				// =================not used??? the outlierTH are set equally??? =============
				float sumGrad2=0; // sumed gradient
				for(int idx=0;idx<patternNum;idx++) //#define patternNum 8
				{	
					// #define patternP staticPattern[8]
					int dx = patternP[idx][0];
					int dy = patternP[idx][1];
					float absgrad = cpt[dx + dy*w[lvl]].tail<2>().squaredNorm(); // pixel in the pattern
					sumGrad2 += absgrad;
				}
				// ================= end of not used =============

				// sumGrad2 not used for derermine the outlier threshold?
//				float gth = setting_outlierTH * (sqrtf(sumGrad2)+setting_outlierTHSumComponent);
//				pl[nl].outlierTH = patternNum*gth*gth;
//

				pl[nl].outlierTH = patternNum*setting_outlierTH; //float setting_outlierTH = 12*12;

				nl++;
				assert(nl <= npts);
			}
		}

		// num of initialized points of the level
		numPoints[lvl]=nl;
	}

	delete[] statusMap;
	delete[] statusMapB;

	// Pnt* pl = points[lvl]; for these initialized points 
	// use kdtree to find 10 nearest points and 
	// if not fiest level, save the parent and neighbors indices and distance
	makeNN();

	thisToNext=SE3(); // initialize 
	snapped = false; //?
	frameID = snappedAt = 0;

	for(int i=0;i<pyrLevelsUsed;i++)
		dGrads[i].setZero(); //?

}

// set point energy to 0
// set idepth_new to current idepth
// if is the coarest level, for the bad points, try to initialize depth using nn points
void CoarseInitializer::resetPoints(int lvl)
{
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		pts[i].energy.setZero();
		pts[i].idepth_new = pts[i].idepth;

		// coarest level bad points
		if(lvl==pyrLevelsUsed-1 && !pts[i].isGood)
		{
			float snd=0, sn=0;
			for(int n = 0;n<10;n++)
			{
				if(pts[i].neighbours[n] == -1 || !pts[pts[i].neighbours[n]].isGood) continue;
				snd += pts[pts[i].neighbours[n]].iR; // sum of neighbor depth
				sn += 1; // sum of neighbor number
			}

			if(sn > 0)
			{
				pts[i].isGood=true; 
				pts[i].iR = pts[i].idepth = pts[i].idepth_new = snd/sn;
			}
		}
	}
}


void CoarseInitializer::doStep(int lvl, float lambda, Vec8f inc)
{

	const float maxPixelStep = 0.25;
	const float idMaxStep = 1e10;
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		if(!pts[i].isGood) continue;


		float b = JbBuffer[i][8] + JbBuffer[i].head<8>().dot(inc);
		float step = - b * JbBuffer[i][9] / (1+lambda);


		float maxstep = maxPixelStep*pts[i].maxstep;
		if(maxstep > idMaxStep) maxstep=idMaxStep;

		if(step >  maxstep) step = maxstep;
		if(step < -maxstep) step = -maxstep;

		float newIdepth = pts[i].idepth + step;
		if(newIdepth < 1e-3 ) newIdepth = 1e-3;
		if(newIdepth > 50) newIdepth = 50;
		pts[i].idepth_new = newIdepth;
	}

}
void CoarseInitializer::applyStep(int lvl)
{
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		if(!pts[i].isGood)
		{
			pts[i].idepth = pts[i].idepth_new = pts[i].iR;
			continue;
		}
		pts[i].energy = pts[i].energy_new;
		pts[i].isGood = pts[i].isGood_new;
		pts[i].idepth = pts[i].idepth_new;
		pts[i].lastHessian = pts[i].lastHessian_new;
	}
	std::swap<Vec10f*>(JbBuffer, JbBuffer_new);
}

void CoarseInitializer::makeK(CalibHessian* HCalib)
{
	w[0] = wG[0];
	h[0] = hG[0];

	fx[0] = HCalib->fxl();
	fy[0] = HCalib->fyl();
	cx[0] = HCalib->cxl();
	cy[0] = HCalib->cyl();

	for (int level = 1; level < pyrLevelsUsed; ++ level)
	{
		w[level] = w[0] >> level;
		h[level] = h[0] >> level;
		fx[level] = fx[level-1] * 0.5;
		fy[level] = fy[level-1] * 0.5;
		cx[level] = (cx[0] + 0.5) / ((int)1<<level) - 0.5;
		cy[level] = (cy[0] + 0.5) / ((int)1<<level) - 0.5;
	}

	for (int level = 0; level < pyrLevelsUsed; ++ level)
	{
		K[level]  << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
		Ki[level] = K[level].inverse();
		fxi[level] = Ki[level](0,0);
		fyi[level] = Ki[level](1,1);
		cxi[level] = Ki[level](0,2);
		cyi[level] = Ki[level](1,2);
	}
}


// use kdtree to find 10 nearest points
// For each point, if it is not on the finest pyramid level, the nearest neighbor at above  level is found. 
// This is termed as the "parent" of the current point.
void CoarseInitializer::makeNN()
{
	const float NNDistFactor=0.05;

	typedef nanoflann::KDTreeSingleIndexAdaptor<
			nanoflann::L2_Simple_Adaptor<float, FLANNPointcloud> ,
			FLANNPointcloud,2> KDTree;

	// build indices
	FLANNPointcloud pcs[PYR_LEVELS];
	KDTree* indexes[PYR_LEVELS];
	for(int i=0;i<pyrLevelsUsed;i++)
	{
		pcs[i] = FLANNPointcloud(numPoints[i], points[i]);
		indexes[i] = new KDTree(2, pcs[i], nanoflann::KDTreeSingleIndexAdaptorParams(5) );
		indexes[i]->buildIndex();
	}

	const int nn=10;

	// find NN & parents
	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
	{
		Pnt* pts = points[lvl];
		int npts = numPoints[lvl];

		int ret_index[nn];
		float ret_dist[nn];
		nanoflann::KNNResultSet<float, int, int> resultSet(nn);
		nanoflann::KNNResultSet<float, int, int> resultSet1(1);

		for(int i=0;i<npts;i++)
		{
			//resultSet.init(pts[i].neighbours, pts[i].neighboursDist );
			resultSet.init(ret_index, ret_dist);
			Vec2f pt = Vec2f(pts[i].u,pts[i].v);
			indexes[lvl]->findNeighbors(resultSet, (float*)&pt, nanoflann::SearchParams());
			int myidx=0;
			float sumDF = 0;
			for(int k=0;k<nn;k++)
			{
				pts[i].neighbours[myidx]=ret_index[k];
				float df = expf(-ret_dist[k]*NNDistFactor);
				sumDF += df;
				pts[i].neighboursDist[myidx]=df;
				assert(ret_index[k]>=0 && ret_index[k] < npts);
				myidx++;
			}
			for(int k=0;k<nn;k++)
				pts[i].neighboursDist[k] *= 10/sumDF; // consider weight only?


			if(lvl < pyrLevelsUsed-1 )
			{
				resultSet1.init(ret_index, ret_dist);
				pt = pt*0.5f-Vec2f(0.25f,0.25f);
				indexes[lvl+1]->findNeighbors(resultSet1, (float*)&pt, nanoflann::SearchParams());

				pts[i].parent = ret_index[0];
				pts[i].parentDist = expf(-ret_dist[0]*NNDistFactor);

				assert(ret_index[0]>=0 && ret_index[0] < numPoints[lvl+1]);
			}
			else
			{
				pts[i].parent = -1;
				pts[i].parentDist = -1;
			}
		}
	}



	// done.

	for(int i=0;i<pyrLevelsUsed;i++)
		delete indexes[i];
}
}

