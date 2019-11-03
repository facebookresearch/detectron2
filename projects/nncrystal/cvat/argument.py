def cvat_args(parser):
    parser.add_argument("--cvat_base")
    parser.add_argument("--cvat_host", default="http://localhost:8080")
    parser.add_argument("--cvat_username")
    parser.add_argument("--cvat_password")
    return parser