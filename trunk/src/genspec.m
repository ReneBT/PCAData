function genspec( filename )
% GENSPEC generates a library of simulated pure fatty acid spectra based on
% Rene Beattie's PhD work.
%
% 'C:\A Rene Work\Algo\Jupyter\PCA\PCAData\trunk\data\GCdata.xlsx'
[GCdata,GC_Txt] = xlsread(filename);
% The FAs measured for each species was different depending on what was
% expected to be detectable, therefore FA that were not measured. To create
% an interpolated estimate we will use a low dimensional PCA to estimate
% what the concentration may have been if measured
GCdata0 = GCdata;
GCdata0(isnan(GCdata)) = 0;
[U,sv,L] = svd(GCdata0);
GCdataRecon = U(:,1:4)*sv(1:4,1:4)*L(:,1:4)';
GCdataRecon(GCdataRecon<0) = (GCdataRecon(GCdataRecon<0)- ...
    min(GCdataRecon(GCdataRecon<0))*1.1)/10; % replace negative values with 
                %very low concentration proportional to reconstructed value

ButterGC = GCdataRecon(4:91,:);
AdiposeGC = GCdataRecon(92:end,:);
sample_ID =  char(GC_Txt(5:92,1));
sample_ID_Adi = GC_Txt(93:end,1);
AdiposeSpecies = ~cellfun(@isempty,strfind(sample_ID_Adi,'B')) + ~cellfun(@isempty,strfind(sample_ID_Adi,'P'))*2 + ~cellfun(@isempty,strfind(sample_ID_Adi,'C'))*3;

FAproperties.Carbons = GCdata(1,:)';
FAproperties.Olefins = GCdata(2,:)';
FAproperties.Isomer = GCdata(3,:)';
FAproperties.MolarMass = (FAproperties.Carbons-2)*14 - ...
    FAproperties.Olefins*2 + 15 + 45; %all FA are unbranched
% 2 carbons are terminal groups, so N-2 carbons are methylene (MM 14)
% unless olefin (MM 2 less) MM of terminal methyl is 15, that of carboxylic
% acid is 45. Result checked against:
% http://toolbox.foodcomp.info/References/FattyAcids/Anders%20M%C3%B8ller%20%20-%20%20FattyAcids%20Molecular%20Weights%20and%20Conversion%20Factors.pdf
 
load('C:\A Rene Work\Algo\Papers\Applied Spectroscopy\FA spectra.mat','FAtrends')

FAXcal = 1200:1800;
s = repmat(0.75,length(FAtrends),1);
c = s; w = s; i = s;
for iP = 1:size(FAproperties.Carbons,1)
    CL  = FAproperties.Carbons(iP);
    for pk = 1:size(FAtrends,1)
        c(pk) = polyval(FAtrends{pk,1},CL);
        w(pk) = polyval(FAtrends{pk,2},CL).*3;
        i(pk) = polyval(FAtrends{pk,3},CL);
        if FAproperties.Olefins(iP)>0
            c(pk) = c(pk)+ polyval(FAtrends{pk,4},FAproperties.Olefins(iP));
            i(pk) = i(pk)+ polyval(FAtrends{pk,5},FAproperties.Olefins(iP));
            if FAproperties.Isomer(iP)==2 && pk==2
                c(pk) = 1673; % shift peak in trans
            elseif FAproperties.Isomer(iP)==3 && pk==2
                c(pk) = 1650; % shift peak in conjugated
            elseif FAproperties.Isomer(iP)==2 && pk==6
                i(pk) = 0; % remove peak in trans
            end
        end
    end
    c(c<0) = 0; w(w<0) = 1; i(i<0) = 0;
    simFA(:,iP) = simsignal(FAXcal,c,w,i,s);
end

collagenData = xlsread('C:\A Rene Work\Algo\Jupyter\PCA\PCAData\trunk\data\peak parameters collagen and heme.xlsx','collagen');
simFA(:,28) = simsignal(FAXcal,collagenData(:,1),collagenData(:,3),collagenData(:,2),collagenData(:,6));
hemeData = xlsread('C:\A Rene Work\Algo\Jupyter\PCA\PCAData\trunk\data\peak parameters collagen and heme.xlsx','heme');
simFA(:,29)= simsignal(FAXcal,hemeData(:,1),hemeData(:,3),hemeData(:,2),hemeData(:,6));

% generate butter spectra based on the GC measured values. NB this is
% grossly simplifying the situation - it assumes all FA are FAMEs (few are,
% most are triglycerides), that the spectra of each is unaffected by the
% profile (in reality: 
%   1.  intermolecular forces will depend significantly on
%       the profile and will in turn affect a number of bonds, most noteably
%       those in the CH2 scissor region
%   2.  intramolecular forces will depend on constraints imposed on the
%       chain by neighbouring fatty acid chains which depends on the precise
%       profile. This affects many bands including the C-C stretching modes
%       and the CH2 twist.

save('C:\A Rene Work\Algo\Jupyter\PCA\PCAData\trunk\data\FA spectra.mat','simFA','FAproperties','FAXcal')
save('C:\A Rene Work\Algo\Jupyter\PCA\PCAData\trunk\data\AllGC.mat','ButterGC','AdiposeGC','sample_ID','sample_ID_Adi','AdiposeSpecies')



