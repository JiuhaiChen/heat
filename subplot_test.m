make_it_tight = true;
subplot = @(m,n,p) subtightplot (m, n, p, [0.08 0.05], [0.07 0.06], [0.05 0.1]);
if ~make_it_tight,  clear subplot;  end

for s=1:4
     subplot(2,2,s) 
        surf(xxx,yyy,reshape(u1,[32,32]))
        hold on
        view(2)
        set(gca,'FontSize',8)
        if s==1
            title('(a) the 2nd batch')
            ax = gca;
            ax.FontSize = 10;
        elseif s==2
            title('(b) the 3rd batch')
            ax = gca;
            ax.FontSize = 10;
        elseif s==3
            title('(c) the 4th batch')
            ax = gca;
            ax.FontSize = 10;
        else
            title('(d) the 5th batch')
            ax = gca;
            ax.FontSize = 10;
        end
       
end

hp4 = get(subplot(2,2,4),'Position');
colorbar('Position', [hp4(1)+hp4(3)+0.03  hp4(2)  0.02  hp4(2)+hp4(3)*2.1])