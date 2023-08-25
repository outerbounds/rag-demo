from git import Repo
from git.exc import InvalidGitRepositoryError
import os
import tempfile
import numpy as np

class DocumentationExtractor:

    """
    Based on code from here:https://github.com/outerbounds/mf-data-extractors
    """

    def __init__(self, repo_url=None, local_repo_path=None):

        if local_repo_path is not None:
            self.local_repo_path = local_repo_path
        else:
            self.local_repo_path = None
            self.repo_url = repo_url


    @staticmethod
    def filter_files(
        path_to_dir, 
        base_search_path, 
        exclude_paths=None, 
        exclude_files=[],
        considered_extensions=[]
    ):

        """
        Fetch all files in a directory, excluding those in the exclude_paths list.

        :param path_to_dir: Local path to the repository/directory to search.
        :param base_search_path: Subdirectory to search within the repo.
        :param exclude_paths: List of paths to exclude. 
        :param considered_extensions: List of file extensions to consider.
        """
    
        if exclude_paths is None:
            exclude_paths = []

        # Convert exclude paths to absolute paths
        exclude_paths = [
            os.path.join(path_to_dir, exclude_path) 
            for exclude_path in exclude_paths
        ]
        selected_files = []

        path_is_excluded = lambda pth: any(
            [True if x in pth else False for x in exclude_paths]
        )
        extension_is_valid = (
            lambda pth: True
            if len(considered_extensions) == 0
            else any([pth.endswith(x) for x in considered_extensions])
        )
        base_abs_path = os.path.join(path_to_dir, base_search_path)

        for root, _, files in os.walk(base_abs_path):

            # Skip excluded paths
            if path_is_excluded(os.path.abspath(root)):
                continue

            for file in files:
                fp = os.path.join(os.path.abspath(root), file)
                if fp in exclude_paths:
                    continue
                if np.any([
                    file.endswith(exclude_file_ext) 
                    for exclude_file_ext in exclude_files
                ]):
                    continue
                if extension_is_valid(file):
                    selected_files.append((fp, os.path.relpath(fp, base_abs_path)))

        return selected_files

    def extract(
        self,
        base_path,
        ref="master",
        exclude_paths=[],
        exclude_files=[],
        considered_extensions=[],
        parser=None,
    ):

        """
        Extract data from a repository.
        Tries to clone the repository to a temporary directory.
        Checks out the specified ref.
        Applies the parser to each file meeting criteria specified in args passed to self.filter_files.
        """

        assert parser is not None
        assert callable(parser)

        if self.local_repo_path:
            file_paths = self.filter_files(
                path_to_dir=self.local_repo_path,
                base_search_path=base_path,
                exclude_paths=exclude_paths,
                exclude_files=exclude_files,
                considered_extensions=considered_extensions,
            )
            parsed_data = [parser(fp, fn) for fp, fn in file_paths]
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                repo = Repo.clone_from(self.repo_url, tmpdir)
                repo.git.checkout(ref)
                file_paths = self.filter_files(
                    path_to_dir=tmpdir,
                    base_search_path=base_path,
                    exclude_paths=exclude_paths,
                    exclude_files=exclude_files,
                    considered_extensions=considered_extensions,
                )
                parsed_data = [parser(fp, fn) for fp, fn in file_paths]

        return parsed_data

class RepoNotFoundException(Exception):
    
    def __init__(self, repo_name):
        self.repo_name = repo_name
        self.message = "Could not find `local_repo_path=%s`" % repo_name
        super().__init__(self.message)


def ensure_repo_exists_locally(local_repo_path=None, ref='master'):

    from git import Repo
    from git.exc import NoSuchPathError
    import os

    local_repo_path = os.path.expanduser(local_repo_path)
    try:
        repo = Repo(local_repo_path)
    except NoSuchPathError:
        raise RepoNotFoundException (f'\n\n{local_repo_path} is not a valid git repository path. It should look like: /path/to/repo. If you are trying to clone a remote repository it should be structured like git@github.com:<ORG or USERNAME>/<REPO NAME>.git')
    except InvalidGitRepositoryError:
        raise RepoNotFoundException (f'\n\n{local_repo_path} is not a valid git repository path. It should look like: /path/to/repo. If you are trying to clone a remote repository it should be structured like git@github.com:<ORG or USERNAME>/<REPO NAME>.git')

    repo.git.checkout(ref)
    assert local_repo_path is not None
    return local_repo_path