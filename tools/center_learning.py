class test_HVCMLoss(torch.nn.Module):
    '''
    Loss for training GMMs.
    '''
    def __init__(self, args, num_classes=64, feat_dim=[32, 256], use_gpu=True, decom=True):
        super(test_HVCMLoss, self).__init__()
        self.num_classes = num_classes

        self.feat_dim = feat_dim

        assert len(feat_dim) == 2
        self.centers_shape = feat_dim.copy()
        self.centers_shape.insert(0, num_classes)
        self.covars_shape = self.centers_shape.copy()
        self.covars_shape.append(feat_dim[1])


        if not decom:
            self.centers_shape = [num_classes, feat_dim[0] * feat_dim[1]]
            self.covars_shape = [num_classes, feat_dim[0] * feat_dim[1], feat_dim[0] * feat_dim[1]]

        if use_gpu:
            self.centers = torch.nn.Parameter(torch.randn(self.centers_shape).cuda())
            self.gmm_weights = torch.nn.Parameter(torch.softmax(torch.randn(num_classes, feat_dim[0]), dim=-1).cuda(), requires_grad=False)
        else:
            self.centers = torch.nn.Parameter(torch.randn(self.centers_shape))
            self.gmm_weights = torch.nn.Parameter(torch.softmax(torch.randn(num_classes, feat_dim[0]), dim=-1), requires_grad=False)

        self.args = args
        # if self.training:
        #     rootpath = '/home/jnx/code/Object_Detection/SSLAD-2D/labeled/train/'
        #     annpath = "/home/jnx/code/Object_Detection/SSLAD-2D/labeled/annotations/instance_train.json"
        #     dataset = SODA_dataset(root=rootpath, annotation=annpath, transforms=get_transform())


    def forward(self, gmm_weights, x, labels, args, training):
        """
        Args:
            gmm_weights: weights of Gaussian conpoments with shape (num_kernel, 1)
            x: feature matrix with shape (batch_size, feat_dim)
            labels: ground truth labels with shape (num_classes)
            Args: an argumentparser.
        """
        self.training = training
        if self.training:

            x = x.reshape(-1, self.feat_dim[0], self.feat_dim[1])#(4096,32,256)
            gmm_weights = gmm_weights.reshape(-1, self.feat_dim[0]) #(4096,32)
            labels = torch.cat(labels, dim=0) #(4096,)

            assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

            total_loss = 0

            center = self.centers[labels]# centers here transfer to 4096,32,256

            '''loss for center'''
            kl_x = F.kl_div(F.log_softmax(x,dim=-1), F.softmax(center.clone().detach(), dim=-1), reduction='none').sum(-1) * self.args.alpha
            kl_ce = F.kl_div(F.log_softmax(center, dim=-1), F.softmax(x.clone().detach(),dim=-1), reduction='none').sum(-1) * self.args.beta

            '''loss for gmm weights'''
            loss = (kl_x.clone().detach() * F.softmax(gmm_weights, dim=-1)).mean()
            total_loss += loss

            js = (kl_ce + kl_x)

            loss = torch.clamp(js, min=1e-5, max=1e+5).mean()
            total_loss += loss

            for cls in labels.unique():
                self.gmm_weights[cls] = F.softmax(gmm_weights[labels==cls], dim=-1).detach().mean(0) * self.args.gamma + \
                                        self.gmm_weights[cls] * (1 - self.args.gamma)

            means = self.centers
            if use_cuda:
                means = means.cuda()


            mahas = self.get_L2_score(means,  # covs_inv,
                                      gmm_weights,
                                      x.reshape(-1, self.args.num_kernel, self.args.out_dim // self.args.num_kernel))

            return total_loss, mahas

        else:

            #print('==> Preparing model..')

            #means = get_gaussian(self.args.pretrained_weights)
            means = self.centers

            # print(means.shape, means.dtype)
            #assert 1==2

            if use_cuda:
                means = means.cuda()


            mahas = self.get_L2_score(means,  # covs_inv,
                                          gmm_weights, x.reshape(-1, self.args.num_kernel, self.args.out_dim // self.args.num_kernel))

            #mahas = F.normalize(input=mahas,dim=-1)
            # print('before softmax',mahas[0])
            # temperature = 5.0
            # mahas = F.softmax(-mahas/temperature,dim=-1)
            # print('after softmax', mahas[0])
            return mahas



    def get_L2_score(self, mu, gmm_weights, x):
        '''
        Args:
            mu: centers of gmm of all classes with shape (classes, kernels, dimensions)
            cov_inv: The inverse matrix of cov which has shape (classes, kernels, dimensions, dimensions)
            gmm_weights: weights of gmm with shape (classes, kernels)
            x: features of input with shape (num_samples, kernels, dimensions)
        '''
        cls, kers, dims = mu.shape
        num = x.shape[0]
        # x = x.squeeze()

        for i in range(cls):
            mu_ = mu[i:i + 1].expand(num, kers, dims)

            x = x.reshape(-1, 1, dims).double()
            mu_ = mu_.reshape(-1, 1, dims).double()

            maha = x - mu_
            maha = 0.5 * torch.bmm(maha, (x - mu_).permute(0, 2, 1)).reshape(num, 1, kers)
            maha = (maha * gmm_weights[i]).sum(-1)
            if i == 0:
                mahas = maha
            else:
                mahas = torch.cat([mahas, maha], dim=1)#batch_size, cls
        #min_maha, result = mahas.min(1)
        return mahas
        #return min_maha, result

    def ood_maha(self, args):
        print('==> Preparing model..')
        # model = torchvision_models.__dict__[args.arch]()
        # embed_dim = model.fc.weight.shape[1]
        # model = utils.MultiCropWrapper(
        #     model,
        #     utils.DINOHead(embed_dim, args.out_dim, False, num_kernel=args.num_kernel),
        # )

        model = test_fasterrcnn_resnet50_fpn(args=args)

        print('==> Preparing GMMs..')
        _, gmm_weights = load_pretrained_weights(model, args.pretrained_weights, 'teacher')
        means = get_gaussian(args.pretrained_weights)

        if use_cuda:
            model.cuda()
            means = means.cuda()
            # covs_inv = covs_inv.cuda()

        model.train(mode=True)  # evevaluation 部分代码代码没改，先用training部分代码顶着

        transform = transforms.Compose([
            transforms.Resize((280, 280)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        print('==> Preparing InD dataset..')

        # dataset = Imagenet('train', args.data_path, i, transform)
        # 这部分dataset要改成eval路径
        rootpath = '/home/jnx/code/Object_Detection/SSLAD-2D/labeled/train/'
        annpath = "/home/jnx/code/Object_Detection/SSLAD-2D/labeled/annotations/instance_train.json"
        dataset = SODA_dataset(root=rootpath, annotation=annpath, transforms=get_transform())
        dataset.data_clean()
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size_per_gpu,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
            sampler=sampler
        )
        result_in = self.get_results(model, dataloader, means, gmm_weights, args.out_dim, args.num_kernel)