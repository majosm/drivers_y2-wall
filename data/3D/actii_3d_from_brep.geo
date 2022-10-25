SetFactory("OpenCASCADE");
surface_vector[] = ShapeFromFile("actii-3d.brep");
//Merge "actii-3d.brep";

// Millimeters to meters
Mesh.ScalingFactor = 0.001;

If(Exists(size))
    basesize=size;
Else
    basesize=6.4;
EndIf

If(Exists(blratio))
    boundratio=blratio;
Else
    boundratio=1.0;
EndIf

If(Exists(blratiocavity))
    boundratiocavity=blratiocavity;
Else
    boundratiocavity=1.0;
EndIf

If(Exists(blratioinjector))
    boundratioinjector=blratioinjector;
Else
    boundratioinjector=1.0;
EndIf

If(Exists(blratiosample))
    boundratiosample=blratiosample;
Else
    boundratiosample=1.0;
EndIf

If(Exists(blratiosurround))
    boundratiosurround=blratiosurround;
Else
    boundratiosurround=1.0;
EndIf

If(Exists(injectorfac))
    injector_factor=injectorfac;
Else
    injector_factor=5.0;
EndIf

If(Exists(shearfac))
    shear_factor=shearfac;
Else
    shear_factor=1.0;
EndIf

If(Exists(isofac))
    iso_factor=isofac;
Else
    iso_factor=1.0;
EndIf

If(Exists(cavityfac))
    cavity_factor=cavityfac;
Else
    cavity_factor=1.0;
EndIf

// horizontal injection
cavityAngle=45;
inj_h=4.;  // height of injector (bottom) from floor
inj_d=1.59; // diameter of injector
inj_l = 20; // length of injector

bigsize = basesize*4;     // the biggest mesh size 
inletsize = basesize*2;   // background mesh size upstream of the nozzle
isosize = basesize/iso_factor;       // background mesh size in the isolator
nozzlesize = basesize/12;       // background mesh size in the nozzle
cavitysize = basesize/cavity_factor; // background mesh size in the cavity region
shearsize = isosize/shear_factor; // background mesh size in the shear region
samplesize = basesize/2;       // background mesh size in the sample
injectorsize = inj_d/injector_factor; // background mesh size in the injector region

Printf("basesize = %f", basesize);
Printf("inletsize = %f", inletsize);
Printf("isosize = %f", isosize);
Printf("nozzlesize = %f", nozzlesize);
Printf("cavitysize = %f", cavitysize);
Printf("shearsize = %f", shearsize);
Printf("injectorsize = %f", injectorsize);
Printf("samplesize = %f", samplesize);
Printf("boundratio = %f", boundratio);
Printf("boundratiocavity = %f", boundratiocavity);
Printf("boundratioinjector = %f", boundratioinjector);
Printf("boundratiosample = %f", boundratiosample);
Printf("boundratiosurround = %f", boundratiosurround);


Physical Volume('fluid') = {3};
Physical Volume('wall_insert') = {2};
Physical Volume('wall_surround') = {1};

Physical Surface("inflow") = {18}; // inlet
Physical Surface("outflow") = {27}; // outlet
Physical Surface("injection") = {35}; // injection
Physical Surface("flow") = {18, 27, 35}; // injection
Physical Surface('isothermal_wall') = {19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34};
Physical Surface('wall_farfield') = {
    1, // wall surround bottom
    2, // wall surround aft
    3, // wall surround back
    4  // wall surround fore
};

// Create distance field from surfaces for wall meshing, excludes cavity, injector
Field[1] = Distance;
Field[1].SurfacesList = {
19, // upstream top
20, // upstream bottom
21, // aft wall
22, // fore wall
23, // upstream top slant
24, // upstream bottom slant
25, // nozzle top
26, // isolator/combustor top
28, // combustor bottom
32, // isolator bottom
33  // pre-nozzle bottom
};
Field[1].Sampling = 1000;
////
//Create threshold field that varrries element size near boundaries
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = isosize / boundratio;
Field[2].SizeMax = isosize;
Field[2].DistMin = 0.02;
Field[2].DistMax = 10;
Field[2].StopAtDistMax = 1;
//
// Create distance field from curves, cavity only
Field[11] = Distance;
Field[11].SurfacesList = {
29, // cavity slant (injection surface)
30, // cavity bottom
31 // cavity front
};
Field[11].Sampling = 1000;

//Create threshold field that varies element size near boundaries
Field[12] = Threshold;
Field[12].InField = 11;
Field[12].SizeMin = cavitysize / boundratiocavity;
Field[12].SizeMax = cavitysize;
Field[12].DistMin = 0.02;
Field[12].DistMax = 10;
Field[12].StopAtDistMax = 1;

// Create distance field from curves, injector only
Field[13] = Distance;
Field[13].SurfacesList = {
34 // injector wall
};
Field[13].Sampling = 1000;
//
//Create threshold field that varrries element size near boundaries
Field[14] = Threshold;
Field[14].InField = 13;
Field[14].SizeMin = injectorsize / boundratioinjector;
Field[14].SizeMax = injectorsize;
Field[14].DistMin = 0.001;
Field[14].DistMax = 1.0;
Field[14].StopAtDistMax = 1;

// Create distance field from curves, inside wall only
Field[15] = Distance;
Field[15].SurfacesList = {
7:15
};
Field[15].Sampling = 1000;

//Create threshold field that varrries element size near boundaries
Field[16] = Threshold;
Field[16].InField = 15;
Field[16].SizeMin = samplesize / boundratiosurround;
Field[16].SizeMax = samplesize;
Field[16].DistMin = 0.2;
Field[16].DistMax = 5;
Field[16].StopAtDistMax = 1;

// Create distance field from curves, sample/fluid interface
Field[17] = Distance;
Field[17].SurfacesList = {
    5,  // surround/fluid slant
    6,  // surround/fluid top
    16, // sample/fluid slant
    17  // sample/fluid top
};
Field[17].Sampling = 1000;

//Create threshold field that varies element size near boundaries
Field[18] = Threshold;
Field[18].InField = 17;
Field[18].SizeMin = cavitysize / boundratiosample;
Field[18].SizeMax = cavitysize;
Field[18].DistMin = 0.2;
Field[18].DistMax = 5;
Field[18].StopAtDistMax = 1;
//
nozzle_start = 270;
nozzle_end = 300;
//  background mesh size in the isolator (downstream of the nozzle)
Field[3] = Box;
Field[3].XMin = nozzle_end;
Field[3].XMax = 1000.0;
Field[3].YMin = -1000.0;
Field[3].YMax = 1000.0;
Field[3].ZMin = -1000.0;
Field[3].ZMax = 1000.0;
Field[3].VIn = isosize;
Field[3].VOut = bigsize;
//
//// background mesh size upstream of the inlet
Field[4] = Box;
Field[4].XMin = 0.;
Field[4].XMax = nozzle_start;
Field[4].YMin = -1000.0;
Field[4].YMax = 1000.0;
Field[4].ZMin = -1000.0;
Field[4].ZMax = 1000.0;
Field[4].VIn = inletsize;
Field[4].VOut = bigsize;
//
// background mesh size in the nozzle throat
Field[5] = Box;
Field[5].XMin = nozzle_start;
Field[5].XMax = nozzle_end;
Field[5].YMin = -1000.0;
Field[5].YMax = 1000.0;
Field[5].ZMin = -1000.0;
Field[5].ZMax = 1000.0;
Field[5].Thickness = 100;    // interpolate from VIn to Vout over a distance around the box
Field[5].VIn = nozzlesize;
Field[5].VOut = bigsize;
//
// background mesh size in the cavity region
cavity_start = 650;
cavity_end = 742;
Field[6] = Box;
Field[6].XMin = cavity_start;
Field[6].XMax = cavity_end;
Field[6].YMin = -1000.0;
Field[6].YMax = 0.;
Field[6].ZMin = -1000.0;
Field[6].ZMax = 1000.0;
Field[6].Thickness = 100;    // interpolate from VIn to Vout over a distance around the box
Field[6].VIn = cavitysize;
Field[6].VOut = bigsize;
//
// background mesh size in the injection region
injector_start_x = 0.69*1000;
injector_end_x = 0.75*1000;
//injector_start_y = -0.0225*1000;
injector_start_y = -0.018*1000;
injector_end_y = -0.026*1000;
injector_start_z = -3;
injector_end_z = 3;
Field[7] = Box;
Field[7].XMin = injector_start_x;
Field[7].XMax = injector_end_x;
Field[7].YMin = injector_start_y;
Field[7].YMax = injector_end_y;
Field[7].ZMin = injector_start_z;
Field[7].ZMax = injector_end_z;
Field[7].Thickness = 100;    // interpolate from VIn to Vout over a distance around the cylinder
////Field[7] = Cylinder;
////Field[7].XAxis = 1;
////Field[7].YCenter = -0.0225295;
////Field[7].ZCenter = 0.0157;
////Field[7].Radius = 0.003;
Field[7].VIn = injectorsize;
Field[7].VOut = bigsize;

// background mesh size in the sample region
Field[8] = Constant;
Field[8].VolumesList = {1,2};
Field[8].VIn = samplesize;
Field[8].VOut = bigsize;

// background mesh size in the shear region
shear_start_x = 0.65*1000;
shear_end_x = 0.73*1000;
shear_start_y = -0.004*1000;
shear_end_y = -0.01*1000;
shear_start_z = -1000.0;
shear_end_z = 1000.0;
Field[9] = Box;
Field[9].XMin = shear_start_x;
Field[9].XMax = shear_end_x;
Field[9].YMin = shear_start_y;
Field[9].YMax = shear_end_y;
Field[9].ZMin = shear_start_z;
Field[9].ZMax = shear_end_z;
Field[9].Thickness = 100;
Field[9].VIn = shearsize;
Field[9].VOut = bigsize;

// keep the injector boundary spacing in the fluid mesh only
Field[20] = Restrict;
Field[20].VolumesList = {3};
Field[20].InField = 14;

Field[21] = Restrict;
Field[21].VolumesList = {3};
Field[21].InField = 7;

// take the minimum of all defined meshing fields
Field[100] = Min;
//Field[100].FieldsList = {2, 3, 4, 5, 6, 7, 12, 14};
//Field[100].FieldsList = {2, 3, 4, 5, 6, 7, 8, 12, 14, 16, 18, 20, 21};
Field[100].FieldsList = {2, 3, 4, 5, 6, 8, 9, 12, 16, 18, 20, 21};
Background Field = 100;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;


// Delaunay, seems to respect changing mesh sizes better
// Mesh.Algorithm3D = 1;
// Frontal, makes a better looking mesh, will make bigger elements where I don't want them though
// Doesn't repsect the mesh sizing parameters ...
//Mesh.Algorithm3D = 4;
// HXT, re-implemented Delaunay in parallel
Mesh.Algorithm3D = 10;
Mesh.OptimizeNetgen = 1;
Mesh.Smoothing = 100;
